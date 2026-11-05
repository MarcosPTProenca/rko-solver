"""Trial executors of the expanded utility study.

Objective:
- Run one planned trial for either frozen mechanism and return an independently scored row.

Technical approach:
- `UtilityScoring` rebuilds each instance's objective outside the solver: the Knapsack
  shortfall to the exact optimum, the rounded tour length and the shifted Rastrigin value.
- `TemperatureExecutor` drives the repository-only native benchmark with the exported
  packaged offline specification, one worker and the requested budget.
- `AdaptiveExecutor` drives the public Python solver through the packaged parameters, one
  worker, with `progress_schedule` toggled in a copied configuration.

Flow (pseudocode):
1. Look up the instance, build its scorer and run the solver under the profile budget.
2. Verify budget, stop reason, mode and the solver-reported objective against the scorer.
3. Return the row; any mismatch raises and aborts the campaign.

Important criteria:
- Evaluation profiles must end at the evaluation cap and time profiles at the time limit.
- A candidate flag or specification the solver did not apply is an error, not a result.

Inputs/Outputs:
- Input: manifest entries, a planned trial and a solver binary or parameter directories.
- Output: score, feasibility, evaluations, stop reason and timing per trial.
"""

from __future__ import annotations

import json
import math
import numpy as np
import subprocess
import time
from benchmarks.algorithms.sa_specification import SaSpecification
from benchmarks.algorithms.utility_inputs import UtilityInputs
from benchmarks.problems.numeric import rastrigin_objectives
from benchmarks.problems.numeric import tsp_objectives
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from examples.run_kp_cpp_instances import read_kp_instance
from pathlib import Path
from rko_solver import RKO
from rko_solver import Numba
from rko_solver import problems
from typing import Any

EVALUATIONS = 100_000
SECONDS = 10.0
SAFETY_CAP = 600.0
ATOL = 1e-8
RTOL = 1e-10
PROBLEM_FLAG = {"item": "kp", "route": "tsp", "continuous": "rastrigin"}


@dataclass(frozen=True, slots=True)
class Scoring:
    """Hold one instance's independent objective and its public problem object.

    Attributes:
        dimension (int): Number of random keys.
        score (Callable[[Sequence[float]], float]): Independent score, lower is better.
        feasible (Callable[[Sequence[float]], bool]): Independent feasibility of a key vector.
        problem (Any): Object accepted by the public solver.
    """

    dimension: int
    score: Callable[[Sequence[float]], float]
    feasible: Callable[[Sequence[float]], bool]
    problem: Any


class UtilityScoring:
    """Build independent scorers for manifest entries.

    Responsibilities:
    - Recompute item-selection, routing and continuous objectives outside the solver.

    Attributes:
        inputs (UtilityInputs): Factory that provides the routing scorer.
    """

    def __init__(self, inputs: UtilityInputs) -> None:
        """Bind the scorer factory.

        Args:
            inputs (UtilityInputs): Factory that provides the routing scorer.
        """
        self.inputs = inputs

    def for_entry(self, entry: Mapping[str, Any]) -> Scoring:
        """Build the scorer of one manifest entry.

        Args:
            entry (Mapping[str, Any]): Manifest entry.

        Returns:
            Scoring: Independent scorer and public problem.

        Raises:
            ValueError: If the class is unknown or the file is malformed.
            OSError: If the file cannot be read.
        """
        path = Path(entry["path"])
        if entry["class"] == "item":
            return self._item(path, int(entry["reference"]))
        if entry["class"] == "route":
            return self._file_problem(path, self.inputs.route_scorer(path), "route")
        if entry["class"] == "continuous":
            return self._file_problem(path, self._continuous(path), "continuous")
        raise ValueError(f"unknown class {entry['class']!r}")

    @staticmethod
    def _item(path: Path, optimum: int) -> Scoring:
        """Score Knapsack as the shortfall to the exact optimum.

        Args:
            path (Path): KPLIB file.
            optimum (int): Exact optimal profit.

        Returns:
            Scoring: Shortfall scorer with capacity feasibility.

        Raises:
            ValueError: If the file is malformed.
        """
        data = read_kp_instance(path)
        problem = problems.Knapsack(data.benefits, data.weights, data.capacity)

        def feasible(keys: Sequence[float]) -> bool:
            """Check the capacity from the key threshold the decoder uses.

            Args:
                keys (Sequence[float]): Random keys.

            Returns:
                bool: Whether the selected weight fits.
            """
            weight = sum(w for key, w in zip(keys, data.weights, strict=True) if key > 0.5)
            return weight <= data.capacity

        def score(keys: Sequence[float]) -> float:
            """Return optimum minus profit, plus the infeasibility penalty.

            Args:
                keys (Sequence[float]): Random keys.

            Returns:
                float: Integral shortfall.
            """
            return float(problem.evaluate(list(keys))) + optimum

        return Scoring(len(data.weights), score, feasible, problem)

    @staticmethod
    def _continuous(path: Path) -> Callable[[Sequence[float]], float]:
        """Build the NumPy shifted Rastrigin scorer.

        Args:
            path (Path): Shift file.

        Returns:
            Callable[[Sequence[float]], float]: Objective of a key vector.

        Raises:
            OSError: If the file cannot be read.
        """
        _, _, blocked = rastrigin_objectives(path)
        return lambda keys: float(blocked(np.asarray([list(keys)], dtype=np.float64))[0])

    @staticmethod
    def _file_problem(path: Path, score: Callable[[Sequence[float]], float], kind: str) -> Scoring:
        """Wrap a file-backed objective with its compiled public problem.

        Args:
            path (Path): Instance file.
            score (Callable[[Sequence[float]], float]): Independent scorer.
            kind (str): `route` or `continuous`.

        Returns:
            Scoring: Scorer with an always-feasible key space.

        Raises:
            OSError: If the file cannot be read.
        """
        build = tsp_objectives if kind == "route" else rastrigin_objectives
        size, compiled, _ = build(path)
        return Scoring(size, score, lambda keys: True, Numba(compiled, size))


class _Verifier:
    """Check a solver outcome against the profile budget and the independent scorer.

    Responsibilities:
    - Enforce the stop reason and evaluation cap of the profile.
    - Compare the solver's reported objective with the independent score.
    """

    @staticmethod
    def check(profile: str, stop: str, evaluations: int, cap: int | None) -> None:
        """Require the profile's intended stop.

        Args:
            profile (str): `evaluations` or `time`.
            stop (str): Reported stop reason.
            evaluations (int): Completed evaluations.
            cap (int | None): Per-algorithm evaluation cap, if the profile has one.

        Raises:
            RuntimeError: If the run did not end as its profile requires.
        """
        expected = "max_evaluations" if profile == "evaluations" else "time_limit"
        if stop != expected or evaluations <= 0:
            raise RuntimeError(f"{profile} profile ended with {stop!r} after {evaluations}")
        if cap is not None and profile == "evaluations" and evaluations < cap:
            raise RuntimeError(f"evaluation profile stopped early at {evaluations}")

    @staticmethod
    def objective(kind: str, reported: float, score: float, reference: int | None) -> None:
        """Require the solver-reported objective to match the independent score.

        Args:
            kind (str): Problem class.
            reported (float): Objective reported by the solver.
            score (float): Independent score.
            reference (int | None): Exact optimum for item selection.

        Raises:
            RuntimeError: If they disagree beyond the reproducibility floor.
        """
        independent = score - reference if kind == "item" and reference else score
        if not math.isclose(reported, independent, rel_tol=RTOL, abs_tol=ATOL):
            raise RuntimeError(f"reported {reported} differs from independent {independent}")


class TemperatureExecutor:
    """Run offline simulated annealing trials through the repository-only native benchmark.

    Responsibilities:
    - Pass the exported packaged specification and one worker to the executable.
    - Verify the parameter echo, calibration flag, budget and objective.

    Attributes:
        binary (Path): Native benchmark executable.
        entries (dict[str, Mapping[str, Any]]): Manifest entries by instance name.
        scoring (UtilityScoring): Independent scorer factory.
    """

    def __init__(
        self,
        binary: Path,
        entries: Sequence[Mapping[str, Any]],
        scoring: UtilityScoring,
        budget: tuple[float, int] = (SECONDS, EVALUATIONS),
    ) -> None:
        """Index the manifest and bind the executable.

        Args:
            binary (Path): Native benchmark executable.
            entries (Sequence[Mapping[str, Any]]): Manifest entries.
            scoring (UtilityScoring): Independent scorer factory.
            budget (tuple[float, int]): Search seconds and per-algorithm evaluation cap of
                the time and evaluation profiles; the registered values by default.
        """
        self.binary = binary.resolve()
        self.entries = {entry["instance"]: entry for entry in entries}
        self.scoring = scoring
        self.budget = budget

    def command(self, trial: Mapping[str, Any], dimension: int) -> list[str]:
        """Build the benchmark command line of one trial.

        Args:
            trial (Mapping[str, Any]): Planned trial.
            dimension (int): Key count, for the packaged specification.

        Returns:
            list[str]: Argument vector.
        """
        entry = self.entries[trial["instance"]]
        timed = trial["profile"] == "time"
        command = [
            str(self.binary),
            entry["path"],
            str(self.budget[0] if timed else SAFETY_CAP),
            "--problem",
            PROBLEM_FLAG[entry["class"]],
            "--algo",
            "SA",
            "--fixed",
            "--threads",
            "1",
            "--seed",
            str(trial["seed"]),
            *SaSpecification.for_dimension(dimension).native_arguments(),
        ]
        if not timed:
            command += ["--evaluations", str(self.budget[1])]
        return command + (["--sa-t0-auto"] if trial["arm"] == "candidate" else [])

    def __call__(self, trial: Mapping[str, Any]) -> dict[str, Any]:
        """Run and verify one trial.

        Args:
            trial (Mapping[str, Any]): Planned trial.

        Returns:
            dict[str, Any]: Row with score, feasibility, evaluations and timing.

        Raises:
            RuntimeError: If the process fails or its output violates the plan.
            subprocess.TimeoutExpired: If the safety timeout expires.
        """
        entry = self.entries[trial["instance"]]
        scoring = self.scoring.for_entry(entry)
        started = time.monotonic()
        process = subprocess.run(
            self.command(trial, scoring.dimension),
            capture_output=True,
            text=True,
            timeout=SAFETY_CAP + 30,
            check=False,
        )
        wall = time.monotonic() - started
        if process.returncode:
            raise RuntimeError(
                f"native trial failed ({process.returncode}): {process.stderr[-500:]}"
            )
        keys, summary = self._parse(process.stdout)
        self._verify(trial, summary, scoring.dimension)
        score = scoring.score(keys)
        _Verifier.objective(entry["class"], summary["cost"], score, entry["reference"])
        return {
            **trial,
            "score": score,
            "feasible": scoring.feasible(keys),
            "evaluations": summary["evaluations"],
            "stop": summary["stops"][0],
            "search_seconds": summary["elapsed"],
            "wall_seconds": wall,
        }

    @staticmethod
    def _parse(stdout: str) -> tuple[list[float], dict[str, Any]]:
        """Read the key vector and the summary line.

        Args:
            stdout (str): Native output.

        Returns:
            tuple[list[float], dict[str, Any]]: Keys and summary.

        Raises:
            RuntimeError: If the result line is missing.
        """
        lines = stdout.splitlines()
        marker = [line for line in lines if line.startswith("RKO_BENCH_RESULT ")]
        if not marker:
            raise RuntimeError("native output has no result line")
        return json.loads(marker[0].removeprefix("RKO_BENCH_RESULT "))["keys"], json.loads(
            lines[-1]
        )

    def _verify(self, trial: Mapping[str, Any], summary: Mapping[str, Any], dimension: int) -> None:
        """Check mode, specification echo, workers and budget.

        Args:
            trial (Mapping[str, Any]): Planned trial.
            summary (Mapping[str, Any]): Native summary.
            dimension (int): Key count, for the packaged specification.

        Raises:
            RuntimeError: If the executable did not apply the planned configuration.
        """
        SaSpecification.for_dimension(dimension).verify(summary["sa_parameters"] or ())
        candidate = trial["arm"] == "candidate"
        if summary["sa_t0_auto"] is not candidate or summary["adaptive"] or summary["workers"] != 1:
            raise RuntimeError("native run does not match the planned mode")
        _Verifier.check(
            trial["profile"], summary["stops"][0], summary["evaluations"], self.budget[1]
        )


class AdaptiveExecutor:
    """Run adaptive-portfolio trials through the public solver and packaged parameters.

    Responsibilities:
    - Select the schedule through a copied configuration directory per arm.
    - Verify one worker, the profile budget and the independent objective.

    Attributes:
        entries (dict[str, Mapping[str, Any]]): Manifest entries by instance name.
        scoring (UtilityScoring): Independent scorer factory.
        params (Mapping[str, Path]): Parameter directory per arm.
    """

    def __init__(
        self,
        entries: Sequence[Mapping[str, Any]],
        scoring: UtilityScoring,
        params: Mapping[str, Path],
        budget: tuple[float, int] = (SECONDS, EVALUATIONS),
    ) -> None:
        """Index the manifest.

        Args:
            entries (Sequence[Mapping[str, Any]]): Manifest entries.
            scoring (UtilityScoring): Independent scorer factory.
            params (Mapping[str, Path]): Parameter directory per arm.
            budget (tuple[float, int]): Search seconds and per-algorithm evaluation cap of
                the time and evaluation profiles; the registered values by default.
        """
        self.entries = {entry["instance"]: entry for entry in entries}
        self.scoring = scoring
        self.params = params
        self.budget = budget

    def __call__(self, trial: Mapping[str, Any]) -> dict[str, Any]:
        """Run and verify one trial.

        Args:
            trial (Mapping[str, Any]): Planned trial.

        Returns:
            dict[str, Any]: Row with score, feasibility, evaluations and timing.

        Raises:
            RuntimeError: If the run does not end as its profile requires.
        """
        entry = self.entries[trial["instance"]]
        scoring = self.scoring.for_entry(entry)
        timed = trial["profile"] == "time"
        started = time.monotonic()
        result = RKO(
            scoring.problem,
            engine="rust",
            random_state=trial["seed"],
            params_dir=self.params[trial["arm"]],
            n_jobs=1,
            max_evaluations=None if timed else self.budget[1],
        ).solve(self.budget[0] if timed else SAFETY_CAP)
        wall = time.monotonic() - started
        _Verifier.check(trial["profile"], result.stop_reason, result.n_evaluations, self.budget[1])
        score = scoring.score(result.keys)
        _Verifier.objective(entry["class"], result.cost, score, entry["reference"])
        return {
            **trial,
            "score": score,
            "feasible": scoring.feasible(result.keys),
            "evaluations": result.n_evaluations,
            "stop": result.stop_reason,
            "search_seconds": result.elapsed_seconds,
            "wall_seconds": wall,
        }
