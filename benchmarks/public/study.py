"""Public-instance benchmark: run the solver arms on verified instances and append JSONL rows.

Objective:
- Measure how close each solver arm gets to the reference value of public TSP and knapsack
  instances, at fixed wall-clock budgets, over several seeds.

Technical approach:
- Load instances through `PublicSources` (SHA-256 verified) and `PublicParsers`.
- Each run builds a fresh `RKO` solver: the Rust arms use a compiled Numba objective; the Python arm
  uses picklable callbacks in process mode. Only the returned keys are trusted; objective and
  feasibility are recomputed by `PublicProblem.check`.
- Rows are appended to the output file as they finish; `--resume` skips rows already present.

Flow (pseudocode):
1. Load the requested families and select instances (all, or one per family with `--pilot`).
2. For each instance, arm, budget and seed not yet recorded: solve, check, append one row.
3. Write nothing else; reports are produced by `benchmarks.public.report`.

Important criteria:
- Arms: rust-default, rust-SA, rust-BRKGA, rust-ILS (Rust engine) and python-process-default.
- Search budgets default to 10 and 60 seconds; the solver excludes startup from this budget.
- Row wall time includes solver construction and startup, but excludes shared objective preparation.

Inputs/Outputs:
- Input: cache directory, output JSONL, families, budgets, seeds, pilot flag.
- Output: JSONL rows with the fields listed in `ROW_FIELDS`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numpy as np
import platform
import subprocess
import sys
import time
from .parsers import MdkpInstance
from .parsers import PublicParsers
from .parsers import TspInstance
from .problems import KnapsackObjective
from .problems import PublicProblem
from .problems import TspObjective
from .problems import identity
from .sources import PublicSources
from benchmarks.problems.knapsack_utils import exact_profit
from collections.abc import Callable
from examples.run_kp_cpp_instances import read_kp_instance
from itertools import product
from pathlib import Path
from rko_solver import RKO
from rko_solver import Callbacks
from rko_solver import Numba
from typing import Any

ARMS = {
    "rust-default": ("rust", None, None),
    "rust-SA": ("rust", ("SA",), None),
    "rust-BRKGA": ("rust", ("BRKGA",), None),
    "rust-ILS": ("rust", ("ILS",), None),
    "python-process-default": ("python", None, "process"),
}
SWAP_ARMS = {
    "rust-default+delta": ("rust", None, None),
    "rust-SA+delta": ("rust", ("SA",), None),
    "rust-ILS+delta": ("rust", ("ILS",), None),
}
ALL_ARMS = {**ARMS, **SWAP_ARMS}
FAMILIES = ("tsplib", "kplib", "orlib-mknap1", "orlib-mknapcb1")
ROW_FIELDS = (
    "family",
    "instance",
    "n",
    "arm",
    "engine",
    "algorithms",
    "seed",
    "seconds",
    "wall_seconds",
    "evaluations",
    "delta_evaluations",
    "stop_reason",
    "objective",
    "solver_cost",
    "checked_cost",
    "feasible",
    "reference",
    "gap",
    "host",
    "commit",
    "benchmark_sha256",
    "schema_version",
)
MKNAPCB_LIMIT = 10


class PublicLoader:
    """Load every requested public family into `PublicProblem` records.

    Responsibilities:
    - Download and verify each source, then parse and attach its reference value.
    """

    @staticmethod
    def load(cache: Path, families: tuple[str, ...]) -> list[PublicProblem]:
        """Return every problem of the requested families in a fixed order.

        Args:
            cache (Path): Root cache directory.
            families (tuple[str, ...]): Families to load.

        Returns:
            list[PublicProblem]: Problems in family order, then source order.
        """
        problems: list[PublicProblem] = []
        for family in families:
            problems.extend(PublicLoader._family(cache, family))
        return problems

    @staticmethod
    def _family(cache: Path, family: str) -> list[PublicProblem]:
        """Dispatch to the fixed dataset loader.

        Args:
            cache (Path): Verified cache root.
            family (str): Dataset group to load.

        Returns:
            list[PublicProblem]: Fixed selection of public instances.

        Raises:
            ValueError: If the family is unknown or downloaded bytes fail verification.
            OSError: If loading fails.
        """
        if family == "tsplib":
            return PublicLoader._tsplib(cache)
        if family == "kplib":
            return PublicLoader._kplib(cache)
        if family == "orlib-mknap1":
            return PublicLoader._orlib(cache, "mknap1.txt", "orlib-mknap1", None)
        if family == "orlib-mknapcb1":
            return PublicLoader._orlib(cache, "mknapcb1.txt", "orlib-mknapcb1", MKNAPCB_LIMIT)
        raise ValueError(f"Unknown family {family}.")

    @staticmethod
    def _tsplib(cache: Path) -> list[PublicProblem]:
        """Attach published optima to parsed TSPLIB matrices.

        Args:
            cache (Path): Verified cache root.

        Returns:
            list[PublicProblem]: The ten selected tour instances.

        Raises:
            ValueError: If verification or parsing fails.
            OSError: If loading fails.
        """
        solutions = PublicParsers.solutions(
            PublicSources.fetch(PublicSources.tsplib()[-1], cache).read_text(encoding="utf-8")
        )
        problems = []
        for source in PublicSources.tsplib()[:-1]:
            path = PublicSources.fetch(source, cache)
            name = path.stem
            instance: TspInstance = PublicParsers.tsplib(name, path.read_text(encoding="utf-8"))
            problems.append(PublicProblem.from_tsp(instance, solutions.get(name)))
        return problems

    @staticmethod
    def _kplib(cache: Path) -> list[PublicProblem]:
        """Load pinned knapsacks with exact dynamic-programming references.

        Args:
            cache (Path): Verified cache root.

        Returns:
            list[PublicProblem]: Fifteen uniquely identified instances.

        Raises:
            ValueError: If verification or parsing fails.
            OSError: If loading fails.
        """
        problems = []
        for source in PublicSources.kplib():
            path = PublicSources.fetch(source, cache)
            data = read_kp_instance(path)
            weights = np.asarray(data.weights, dtype=float).reshape(1, -1)
            objective = KnapsackObjective(
                np.asarray(data.benefits, dtype=float), weights, np.array([float(data.capacity)])
            )
            reference = float(exact_profit(data))
            problems.append(PublicProblem(source.relative, "kplib", objective, reference))
        return problems

    @staticmethod
    def _orlib(cache: Path, filename: str, family: str, limit: int | None) -> list[PublicProblem]:
        """Load the fixed prefix of an OR-Library multidimensional knapsack file.

        Args:
            cache (Path): Verified cache root.
            filename (str): Catalog filename.
            family (str): Dataset group label.
            limit (int | None): Number of leading instances, or all when absent.

        Returns:
            list[PublicProblem]: Parsed instances with declared references where present.

        Raises:
            ValueError: If verification or parsing fails.
            OSError: If loading fails.
        """
        source = next(s for s in PublicSources.orlib() if s.relative.endswith(filename))
        path = PublicSources.fetch(source, cache)
        instances: tuple[MdkpInstance, ...] = PublicParsers.mdkp(
            filename, path.read_text(encoding="utf-8")
        )
        chosen = instances if limit is None else instances[:limit]
        return [PublicProblem.from_mdkp(family, instance) for instance in chosen]


class PublicStudy:
    """Run arms over problems and append rows, skipping rows that already exist.

    Responsibilities:
    - Build the solver for each arm, solve, check the returned keys, and write one row.
    """

    @staticmethod
    def key(row: dict[str, Any]) -> tuple[str, str, int, float]:
        """Identify one experiment within a provenance-checked campaign.

        Args:
            row (dict[str, Any]): A recorded experiment.

        Returns:
            tuple[str, str, int, float]: Instance, arm, seed and search budget.

        Raises:
            KeyError: If required fields are absent.
            ValueError: If numeric fields are malformed.
        """
        return (str(row["instance"]), str(row["arm"]), int(row["seed"]), float(row["seconds"]))

    @staticmethod
    def provenance() -> dict[str, object]:
        """Identify the machine, source revision and benchmark implementation.

        Returns:
            dict[str, object]: Metadata required to resume the same protocol.
        """
        digest = hashlib.sha256()
        for source in sorted(Path(__file__).parent.glob("*.py")):
            digest.update(source.name.encode("utf-8"))
            digest.update(source.read_bytes())
        return {
            "host": platform.node(),
            "commit": PublicStudy.commit(),
            "benchmark_sha256": digest.hexdigest(),
            "schema_version": 1,
        }

    @staticmethod
    def existing(output: Path) -> set[tuple[str, str, int, float]]:
        """Load compatible completed rows without silently repairing corrupt output.

        Args:
            output (Path): Existing campaign output, if present.

        Returns:
            set[tuple[str, str, int, float]]: Completed experiment identities.

        Raises:
            ValueError: If JSON or campaign provenance is incompatible.
            OSError: If the file cannot be read.
        """
        if not output.exists():
            return set()
        identities = set()
        expected = PublicStudy.provenance()
        for number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{output}:{number} is not valid JSON; fix or remove it."
                ) from error
            if any(row.get(field) != value for field, value in expected.items()):
                raise ValueError(f"{output}:{number}: incompatible provenance; use a new output.")
            identities.add(PublicStudy.key(row))
        return identities

    @staticmethod
    def solve(
        problem: PublicProblem,
        arm: str,
        seed: int,
        seconds: float,
        native: Callable[[np.ndarray], float] | None,
        swap: Callable[[np.ndarray, float, int, int], float] | None = None,
    ) -> dict[str, object]:
        """Run one arm on one problem and return its row.

        Args:
            problem (PublicProblem): Instance with objective and reference.
            arm (str): Key of `ARMS`.
            seed (int): Search seed.
            seconds (float): Wall-clock budget.
            native (Callable[[np.ndarray], float] | None): Prepared native objective, if available.
            swap (Callable[[np.ndarray, float, int, int], float] | None): Prepared swap delta for
                `+delta` arms; compiled on demand when absent.

        Returns:
            dict[str, object]: Row with the fields in `ROW_FIELDS`.
        """
        engine, algorithms, _ = ALL_ARMS[arm]
        began = time.perf_counter()
        solver = PublicStudy.build(problem, arm, seed, native, swap)
        result = solver.solve(seconds)
        wall = time.perf_counter() - began
        checked = problem.check(np.asarray(result.keys, dtype=float))
        if not np.isclose(result.cost, checked.cost, rtol=1e-10, atol=1e-8):
            raise ValueError(
                f"{problem.name}: solver cost {result.cost} != checker cost {checked.cost}."
            )
        return {
            "family": problem.family,
            "instance": problem.name,
            "n": problem.n_variables,
            "arm": arm,
            "engine": engine,
            "algorithms": list(algorithms) if algorithms else "default",
            "seed": seed,
            "seconds": seconds,
            "wall_seconds": wall,
            "evaluations": result.n_evaluations or 0,
            "delta_evaluations": sum(search.delta_evaluations for search in result.searches),
            "stop_reason": result.stop_reason,
            "objective": checked.objective,
            "solver_cost": float(result.cost),
            "checked_cost": checked.cost,
            "feasible": checked.feasible,
            "reference": problem.reference,
            "gap": checked.gap,
            **PublicStudy.provenance(),
        }

    @staticmethod
    def build(
        problem: PublicProblem,
        arm: str,
        seed: int,
        native: Callable[[np.ndarray], float] | None,
        swap: Callable[[np.ndarray, float, int, int], float] | None,
    ) -> RKO:
        """Create the solver an arm runs, with its objective and optional swap delta.

        Args:
            problem (PublicProblem): Instance with objective.
            arm (str): Key of `ALL_ARMS`.
            seed (int): Search seed.
            native (Callable[[np.ndarray], float] | None): Prepared native objective, if any.
            swap (Callable[[np.ndarray, float, int, int], float] | None): Prepared swap delta.

        Returns:
            RKO: Solver configured for the arm.
        """
        engine, algorithms, _ = ALL_ARMS[arm]
        names = list(algorithms) if algorithms else None
        if engine == "rust":
            objective_fn = native if native is not None else problem.objective.numba_function()
            problem_spec = Numba(
                objective_fn, problem.n_variables, swap_delta=_delta_for(arm, problem, swap)
            )
            return RKO(problem_spec, engine="rust", algorithms=names, random_state=seed)
        callbacks = Callbacks(identity, problem.objective.cost, problem.n_variables)
        return RKO(
            callbacks, engine="python", execution="process", algorithms=names, random_state=seed
        )

    @staticmethod
    def commit() -> str:
        """Identify the source checkout without changing its state.

        Returns:
            str: Short commit identifier, or 'unknown' outside a repository.
        """
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except OSError, subprocess.CalledProcessError:
            return "unknown"


def _tsp_swap(problem: PublicProblem) -> Callable[[np.ndarray, float, int, int], float]:
    """Return the exact swap delta of a TSP instance.

    Args:
        problem (PublicProblem): Instance whose objective supplies the delta.

    Returns:
        Callable[[np.ndarray, float, int, int], float]: Compiled swap score.

    Raises:
        ValueError: If the instance is not a TSP, so no swap delta exists for it.
    """
    if not isinstance(problem.objective, TspObjective):
        raise ValueError(f"{problem.name}: swap deltas are defined for TSP instances only.")
    return problem.objective.numba_swap_delta()


def _delta_for(
    arm: str, problem: PublicProblem, swap: Callable[[np.ndarray, float, int, int], float] | None
) -> Callable[[np.ndarray, float, int, int], float] | None:
    """Return the swap delta an arm needs, or None for arms without one.

    Args:
        arm (str): Arm key from `ALL_ARMS`.
        problem (PublicProblem): Instance the arm runs on.
        swap (Callable[[np.ndarray, float, int, int], float] | None): Prepared delta, if any.

    Returns:
        Callable[[np.ndarray, float, int, int], float] | None: Delta for `+delta` arms.
    """
    if arm not in SWAP_ARMS:
        return None
    return swap if swap is not None else _tsp_swap(problem)


def _arms_for(problem: PublicProblem, arms: tuple[str, ...]) -> list[str]:
    """Select the arms runnable on an instance; `+delta` arms need a TSP.

    Args:
        problem (PublicProblem): Instance to run.
        arms (tuple[str, ...]): Requested arm keys.

    Returns:
        list[str]: Arms to run on this instance, in request order.
    """
    return [arm for arm in arms if arm not in SWAP_ARMS or problem.family == "tsplib"]


def _prepared_swap(
    problem: PublicProblem, selected: list[str], native: Callable[[np.ndarray], float]
) -> Callable[[np.ndarray, float, int, int], float] | None:
    """Compile and warm the swap delta once when any selected arm needs it.

    Args:
        problem (PublicProblem): Instance to run.
        selected (list[str]): Arms that will run on the instance.
        native (Callable[[np.ndarray], float]): Compiled objective used to warm the delta.

    Returns:
        Callable[[np.ndarray, float, int, int], float] | None: Warm delta, or None.
    """
    if not any(arm in SWAP_ARMS for arm in selected):
        return None
    swap = _tsp_swap(problem)
    probe = np.arange(problem.n_variables, dtype=float) / problem.n_variables
    swap(probe, native(probe), 0, 1)
    return swap


def _selected(problems: list[PublicProblem], pilot: bool) -> list[PublicProblem]:
    """Return every problem, or the first problem of each family when piloting.

    Args:
        problems (list[PublicProblem]): Problems in family order.
        pilot (bool): Whether to keep one instance per family.

    Returns:
        list[PublicProblem]: The selected problems.
    """
    if not pilot:
        return problems
    first: dict[str, PublicProblem] = {}
    for problem in problems:
        first.setdefault(problem.family, problem)
    return list(first.values())


def _run(
    problems: list[PublicProblem],
    seeds: range,
    seconds: tuple[float, ...],
    output: Path,
    arms: tuple[str, ...] = tuple(ARMS),
) -> None:
    """Solve every missing (instance, arm, seed, budget) and append its row to the output.

    Args:
        problems (list[PublicProblem]): Problems to run.
        seeds (range): Search seeds.
        seconds (tuple[float, ...]): Wall-clock budgets.
        output (Path): JSONL file that receives one row per run.
        arms (tuple[str, ...]): Arm keys from `ALL_ARMS` to run; `+delta` arms need a TSP instance.
    """
    done = PublicStudy.existing(output)
    with output.open("a", encoding="utf-8") as sink:
        for problem in problems:
            native = problem.objective.numba_function()
            native(np.zeros(problem.n_variables, dtype=float))
            selected = _arms_for(problem, arms)
            swap = _prepared_swap(problem, selected, native)
            for arm, seed, budget in product(selected, seeds, seconds):
                key = (problem.name, arm, seed, float(budget))
                if key in done:
                    continue
                row = PublicStudy.solve(problem, arm, seed, budget, native, swap)
                sink.write(json.dumps({field: row[field] for field in ROW_FIELDS}) + "\n")
                sink.flush()
                done.add(key)
                print(json.dumps(row), file=sys.stderr, flush=True)


def main() -> None:
    """Parse arguments, load the public families and run the missing rows.

    Raises:
        ValueError: If a budget or seed count is not positive.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "rko-public")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true", help="Skip compatible rows already written."
    )
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=tuple(ALL_ARMS),
        default=tuple(ARMS),
        help="Arms to run; '+delta' arms apply to TSP only.",
    )
    parser.add_argument("--seconds", type=float, nargs="+", default=(10.0, 60.0))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--pilot", action="store_true", help="One instance per family, 2 s, seed 0."
    )
    arguments = parser.parse_args()
    if arguments.output.exists() and not arguments.resume:
        raise FileExistsError("Output already exists; use --resume or a new output path.")
    if arguments.seeds <= 0 or any(s <= 0 for s in arguments.seconds):
        raise ValueError("Seeds and budgets must be positive.")
    problems = _selected(
        PublicLoader.load(arguments.cache_dir, tuple(arguments.families)), arguments.pilot
    )
    if arguments.pilot:
        _run(problems, range(1), (2.0,), arguments.output, tuple(arguments.arms))
    else:
        _run(
            problems,
            range(arguments.seeds),
            tuple(arguments.seconds),
            arguments.output,
            tuple(arguments.arms),
        )


if __name__ == "__main__":
    main()
