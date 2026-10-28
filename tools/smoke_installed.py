"""Exercise a built artifact without importing the source checkout.

Objective:
- Detect missing runtime resources and unusable installed solver entry points, and that the
  example programs and the demo command are not part of the artifact.

Technical approach:
- Run under python -I inside an environment containing only the installed artifact.
- Check filtered adaptive graphs and package contents.
- Check seeded sequential/thread/spawn execution without modifying caller RNG state.

Flow (pseudocode):
1. Locate the installed package and assert expected data and excluded tests.
2. Read packaged parameters and reference instances.
3. Verify class-based results against an independent objective calculation.

Important criteria:
- Use the main guard and importable callbacks for Windows spawn.
- This is an installation contract check, not an optimization-quality benchmark.
- Spawned workers on a loaded runner can take longer than a few tens of milliseconds to start,
  so portfolio searches use a budget that outlasts startup instead of the smallest valid one.

Inputs/Outputs:
- Input: installed wheel or sdist and its runtime dependencies.
- Output: assertions and a success message, or a failing process exit.
"""

import numpy as np
import os
import pickle
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from rko_solver import RKO
from rko_solver import Blocks
from rko_solver import Numba
from rko_solver import PoolInitializationMetadata
from rko_solver import SearchMetadata
from rko_solver import ValidationCheck
from rko_solver import ValidationReport
from rko_solver import assess_result
from rko_solver.components.adaptive_control.q_learning import QLearningController
from rko_solver.explain import DecoderDiagram
from rko_solver.metaheuristics import SA
from rko_solver.parameters.loader import default_params_dir
from rko_solver.problems import Knapsack
from rko_solver.reporting import OptimizationReport

SEARCH_SECONDS = 0.25


@dataclass(frozen=True, slots=True)
class Scheduling:
    """Minimal pickleable scheduling problem used to exercise every execution mode.

    Responsibilities:
    - Decode priority keys into completion times, score tardiness and check the schedule.

    Attributes:
        processing_times (tuple[int, ...]): Positive duration of each job.
        due_dates (tuple[int, ...]): Due date of each job.
    """

    processing_times: tuple[int, ...]
    due_dates: tuple[int, ...]

    @property
    def n_variables(self) -> int:
        """Return one key per job.

        Returns:
            int: Number of jobs.
        """
        return len(self.processing_times)

    def decode(self, keys: np.ndarray) -> tuple[tuple[int, int], ...]:
        """Order jobs by increasing key and accumulate completion times.

        Args:
            keys (np.ndarray): One finite priority key per job.

        Returns:
            tuple[tuple[int, int], ...]: Job index and completion time pairs.

        Raises:
            ValueError: If the keys are not one finite value per job.
        """
        values = np.asarray(keys, dtype=float)
        if values.ndim != 1 or len(values) != self.n_variables or not np.isfinite(values).all():
            raise ValueError("Decoder needs one finite key per job.")
        elapsed = 0
        schedule = []
        for index in np.argsort(values, kind="stable"):
            elapsed += self.processing_times[int(index)]
            schedule.append((int(index), elapsed))
        return tuple(schedule)

    def cost(self, solution: tuple[tuple[int, int], ...]) -> float:
        """Sum positive lateness of a decoded schedule.

        Args:
            solution (tuple[tuple[int, int], ...]): Job index and completion time pairs.

        Returns:
            float: Total tardiness to minimize.
        """
        return float(sum(max(0, done - self.due_dates[job]) for job, done in solution))

    def validate(self, solution: tuple[tuple[int, int], ...]) -> ValidationReport:
        """Check job coverage and contiguous completion times independently.

        Args:
            solution (tuple[tuple[int, int], ...]): Job index and completion time pairs.

        Returns:
            ValidationReport: Declared feasibility of the schedule.
        """
        covered = sorted(job for job, _ in solution) == list(range(self.n_variables))
        elapsed = 0
        contiguous = covered
        for job, done in solution:
            if covered:
                elapsed += self.processing_times[job]
                contiguous = contiguous and done == elapsed
        checks = (
            ValidationCheck(
                "job_coverage", "passed" if covered else "failed", "Every job exactly once."
            ),
            ValidationCheck(
                "completion_times",
                "passed" if contiguous else "failed",
                "Cumulative processing times without idle gaps.",
            ),
        )
        return ValidationReport(checks=checks, feasible=covered and contiguous)


def decode(keys: np.ndarray) -> np.ndarray:
    """Decode keys without changing the vector.

    Args:
        keys (np.ndarray): Candidate vector.

    Returns:
        np.ndarray: Unmodified candidate.
    """
    return keys


def objective(keys: np.ndarray) -> float:
    """Evaluate the installed smoke objective.

    Args:
        keys (np.ndarray): Candidate vector.

    Returns:
        float: Squared Euclidean norm.
    """
    return float(np.dot(keys, keys))


def main() -> None:
    """Validate installed resources and actual public execution.

    Raises:
        AssertionError: If packaging or solver result contracts fail.
        OSError: If installed runtime resources cannot be read.
    """
    package = default_params_dir().parent
    assert not (package / "tests").exists()
    assert (default_params_dir() / "config.toml").is_file()
    assert len(list(default_params_dir().glob("*.toml"))) == 12
    assert not (package / "examples").exists()
    assert shutil.which("rko-solver", path=str(Path(sys.executable).parent)) is None
    assert package.resolve() != Path(__file__).resolve().parents[1] / "src" / "rko_solver"
    controller = QLearningController(
        [[0.0, 1.0], [0.0, 1.0]], state_filter=lambda values: sum(values) <= 1.0
    )
    assert [state.par for state in controller.states] == [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    assert [state.Ai for state in controller.states] == [[0, 1, 2], [0, 1], [0, 2]]
    assert (package / "_native.abi3.so").is_file() or list(package.glob("_native*"))
    knapsack = Knapsack([4, 3, 2], [3, 2, 2], 4)
    native_result = RKO(
        knapsack,
        engine="rust",
        algorithms=["BRKGA", "SA", "ILS"],
        adaptation="adaptive",
        random_state=7,
    ).solve(0.1)
    assert native_result.engine == "rust" and native_result.n_workers == 3
    assert native_result.cost == knapsack.evaluate(list(native_result.keys))
    assert all(report.n_evaluations >= 1 for report in native_result.searches)
    callback_result = RKO(
        decode, objective, 8, algorithms=["SA"], engine="rust", random_state=7
    ).solve(0.1)
    assert np.isclose(callback_result.cost, objective(np.asarray(callback_result.keys)))
    blocks: list[int] = []

    def block_objective(keys: np.ndarray) -> np.ndarray:
        """Score a block of rows with the smoke objective.

        Args:
            keys (np.ndarray): Candidate matrix.

        Returns:
            np.ndarray: Squared norm per row.
        """
        blocks.append(len(keys))
        return np.einsum("ij,ij->i", keys, keys)

    batched_result = RKO(
        Blocks(block_objective, 8, 16), engine="rust", algorithms=["BRKGA", "ILS"], random_state=7
    ).solve(0.1)
    assert max(blocks) > 1 and sum(blocks) == batched_result.n_evaluations
    assert np.isclose(batched_result.cost, objective(np.asarray(batched_result.keys)))
    try:
        import numba
    except ImportError:
        numba = None
    if numba is not None:
        compiled_result = RKO(
            Numba(numba.njit(lambda keys: float(np.sum(keys * keys))), 8),
            engine="rust",
            algorithms=["SA", "ILS"],
            random_state=7,
        ).solve(0.1)
        assert np.isclose(compiled_result.cost, objective(np.asarray(compiled_result.keys)))
    global_rng = pickle.dumps((random.getstate(), np.random.get_state()))
    for execution in ("sequential", "thread", "process"):
        for algorithm, dimension, control in (
            ("MultiStart", 3, 0),
            ("IPR", 64, 0),
            ("GRASP", 1, 0),
            ("GRASP", 1, 1),
            ("GRASP", 4, 0),
            ("GRASP", 4, 1),
            ("SA", 64, 0),
            ("SA", 64, 1),
            ("PSO", 64, 0),
            ("PSO", 64, 1),
        ):
            result = RKO(
                decode,
                objective,
                dimension,
                algorithms=[algorithm],
                control=control,
                hyperparameters={"PSO": {"p_size": 4, "w": 0.5}}
                if algorithm == "PSO" and control == 0
                else None,
                execution=execution,
                random_state=42,
            ).solve(SEARCH_SECONDS)
            keys = np.asarray(result.keys)
            assert len(keys) == dimension
            assert np.all((keys >= 0.0) & (keys < 1.0))
            assert np.isclose(result.cost, objective(keys))
            assert result.n_evaluations >= 1
            assert result.stop_reason == "time_limit"
            assert result.n_workers == 1
            assert result.available_cpus == (
                max(1, os.process_cpu_count() or 1) if execution == "process" else None
            )
            assert len(result.searches) == 1
            report = result.searches[0]
            assert isinstance(report, SearchMetadata)
            assert report.n_evaluations == result.n_evaluations
            assert report.algorithm == algorithm
            assert report.random_state == int(
                np.random.SeedSequence(42).spawn(1)[0].generate_state(1)[0]
            )
            assert dict(result.versions)["numpy"] == np.__version__
            if algorithm == "PSO" and control == 0:
                assert dict(report.parameters)["p_size"] == 4
            assert bool(report.parameter_grid) == (
                algorithm in ("GRASP", "SA", "PSO") and control == 1
            )
            assert pickle.loads(pickle.dumps(result)) == result
        prefilled = RKO(
            decode,
            objective,
            3,
            algorithms=["MultiStart"],
            control=0,
            execution=execution,
            prefill_pool=True,
            pool_size=3,
            random_state=42,
        ).solve(SEARCH_SECONDS)
        assert isinstance(prefilled.initialization, PoolInitializationMetadata)
        assert prefilled.initialization.n_evaluations == 3
        assert prefilled.initialization.pool_size == 3
        assert prefilled.n_evaluations == 3 + sum(
            report.n_evaluations for report in prefilled.searches
        )
        assert pickle.loads(pickle.dumps(prefilled)) == prefilled
    automatic = RKO(
        decode,
        objective,
        64,
        algorithms=["MultiStart", "PSO"],
        control=0,
        hyperparameters={"PSO": {"p_size": 4}},
        random_state=42,
    ).solve(0.2)
    assert automatic.n_jobs is None
    assert automatic.available_cpus == max(1, os.process_cpu_count() or 1)
    assert automatic.n_workers == min(2, automatic.available_cpus)
    assert [report.algorithm for report in automatic.searches] == ["MultiStart", "PSO"]
    assert np.isclose(automatic.cost, objective(np.asarray(automatic.keys)))
    assert pickle.loads(pickle.dumps(automatic)) == automatic
    uneven = RKO(
        decode,
        objective,
        16,
        algorithms=["MultiStart", "PSO", "GA"],
        control=0,
        hyperparameters={"PSO": {"p_size": 4}, "GA": {"size_pop": 12}},
        execution="process",
        n_jobs=2,
        random_state=42,
    ).solve(0.2)
    assert uneven.n_workers == 2
    assert [report.algorithm for report in uneven.searches] == ["MultiStart", "PSO", "GA"]
    assert 0 < uneven.searches[0].time_limit <= 0.1
    assert 0 <= uneven.searches[1].time_limit <= 0.2
    assert all(0 <= report.time_limit <= 0.2 for report in uneven.searches)
    assert uneven.n_evaluations == sum(report.n_evaluations for report in uneven.searches)
    assert np.isclose(uneven.cost, objective(np.asarray(uneven.keys)))
    assert pickle.loads(pickle.dumps(uneven)) == uneven
    bounded = RKO(
        decode,
        objective,
        64,
        algorithms=["MultiStart", "IPR", "PSO"],
        control=0,
        hyperparameters={"PSO": {"p_size": 4}},
        execution="process",
        n_jobs=1,
        random_state=42,
    ).solve(0.1)
    assert len(bounded.keys) == 64
    assert np.isclose(bounded.cost, objective(np.asarray(bounded.keys)))
    assert bounded.n_workers == 1
    assert [report.algorithm for report in bounded.searches] == ["MultiStart", "IPR", "PSO"]
    assert bounded.n_evaluations == sum(report.n_evaluations for report in bounded.searches)
    assert dict(bounded.searches[-1].parameters)["p_size"] == 4
    assert pickle.dumps((random.getstate(), np.random.get_state())) == global_rng
    problem = Scheduling((3, 2, 5, 1), (4, 6, 7, 3))
    for execution in ("sequential", "thread", "process"):
        scheduled = RKO(
            problem.decode,
            problem.cost,
            problem.n_variables,
            algorithms=["SA"],
            adaptation="fixed",
            execution=execution,
            random_state=42,
        ).solve(0.1)
        assert scheduled.cost == problem.cost(problem.decode(np.asarray(scheduled.keys)))
        assert scheduled.n_variables == problem.n_variables
        assessment = assess_result(scheduled, decoder=problem.decode, validator=problem.validate)
        assert assessment.validation.status == "passed"
        assert assessment.validation.feasible is True
        assert assessment.result is scheduled
    standalone_solver = SA(
        problem.decode,
        problem.cost,
        problem.n_variables,
        adaptation="fixed",
        execution="sequential",
        random_state=42,
        hyperparameters={"SA": {"sa_max": 100, "t0": 1000.0}},
    )
    standalone = standalone_solver.solve(time_limit=0.1)
    assert standalone_solver.result_ is standalone
    assert dict(standalone.searches[0].parameters)["sa_max"] == 100
    assert standalone.algorithms == ("SA",)
    assert standalone.cost == problem.cost(problem.decode(np.asarray(standalone.keys)))
    diagram = DecoderDiagram.from_callable(problem.decode)
    assert diagram.origin == "source_derived"
    assert diagram.function_name == "decode"
    assert any(node.kind == "decision" for node in diagram.nodes)
    assert any(node.kind == "error" for node in diagram.nodes)
    graph_report = OptimizationReport(standalone, decoder_diagram=diagram).to_dict()
    assert graph_report["decoder_diagram"] == diagram.to_dict()
    assert "plotly" not in sys.modules
    print(f"Installed artifact passed: {package}")


if __name__ == "__main__":
    main()
