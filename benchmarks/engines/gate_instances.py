"""Instance selection and construction for the non-inferiority gate.

Objective:
- Draw disjoint calibration and test instances from fixed seeds and build each one with an
  independent scoring closure, so the gate can score solver output without trusting the solver.

Technical approach:
- `GateInstance` is a frozen record of a problem, its reference value and a scorer.
- `GateInstances` chooses KPLIB (type, id) pairs round-robin and writes shifted Rastrigin inputs,
  caching downloads and verifying each file by hash.

Flow (pseudocode):
1. Choose per-family identifiers from the split's seed, excluding calibration ids for the test
   split.
2. For Knapsack, download and hash the KPLIB file and read its exact optimum.
3. For Rastrigin, write the shifted instance and compile its objective; the optimum is zero.

Important criteria:
- The test and calibration splits never share a KPLIB id.
- A KPLIB file must be 500 items and at most 64 KiB.

Inputs/Outputs:
- Input: cache directory, split name and instance count.
- Output: an iterator of ready `GateInstance` objects per family.
"""

from __future__ import annotations

import hashlib
import numpy as np
import random
import urllib.request
from benchmarks.engines.cpp_python_knapsack import FAMILIES
from benchmarks.engines.cpp_python_knapsack import KPLIB_RAW
from benchmarks.engines.cpp_python_knapsack import KPLIB_REVISION
from benchmarks.engines.gate_support import ConfigOverrides
from benchmarks.problems.knapsack_utils import exact_profit
from benchmarks.problems.numeric import rastrigin_objectives
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from examples.run_kp_cpp_instances import read_kp_instance
from pathlib import Path
from rko_solver import Numba
from rko_solver import problems
from typing import Any

CALIBRATION_SEED = 20261120
TEST_SEED = 20261111
CALIBRATION_POOL = 20
RASTRIGIN_DIMENSIONS = (100, 300)
EVALUATION_TIME_CAP = 600.0
EVALUATION_JOBS = 1


@dataclass(frozen=True, slots=True)
class Budget:
    """Describe one search budget.

    Attributes:
        label (str): Short name written to the JSONL file, such as `2s` or `evals`.
        time_limit (float): Wall-clock search seconds handed to the solver.
        max_evaluations (int | None): Per-algorithm evaluation cap, or None for time only.
        n_jobs (int | None): Rust worker threads, or None for the engine default.
    """

    label: str
    time_limit: float
    max_evaluations: int | None
    n_jobs: int | None = None

    def solver_options(self, engine: str) -> dict[str, Any]:
        """Build the budget-dependent keyword arguments of the `RKO` constructor.

        Args:
            engine (str): `rust` or `cpp`; the C++ engine rejects `n_jobs`, so it never gets it.

        Returns:
            dict[str, Any]: `max_evaluations` and, for Rust only, `n_jobs`.
        """
        options: dict[str, Any] = {"max_evaluations": self.max_evaluations}
        if self.n_jobs is not None and engine == "rust":
            options["n_jobs"] = self.n_jobs
        return options


@dataclass(frozen=True, slots=True)
class GateInstance:
    """Hold one benchmark instance with its independent scoring.

    Attributes:
        family (str): Analysis family, such as `knapsack` or `rastrigin-100`.
        name (str): Stable instance identifier.
        sha256 (str): Digest of the instance file.
        reference (float): Reference value: the exact optimum profit or zero.
        problem (Any): Problem object accepted by `RKO`.
        score (Callable[[Sequence[float]], float]): Absolute gap of a key vector to the reference.
    """

    family: str
    name: str
    sha256: str
    reference: float
    problem: Any
    score: Callable[[Sequence[float]], float]


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Hold the run options of one gate invocation.

    Attributes:
        candidate (str): Candidate `engine[/ALGO+ALGO]` spec.
        baseline (str | None): Baseline spec, a JSONL path, or None to only record the candidate.
        families (Sequence[str]): `knapsack`, `rastrigin` (both sizes) or a sized family.
        budgets (Sequence[Budget]): Budgets to run.
        replicates (int): Replicates per instance and budget.
        output (Path): JSONL destination.
        candidate_options (Sequence[str]): `KEY=VALUE` overrides for the candidate.
        baseline_options (Sequence[str]): `KEY=VALUE` overrides for a live baseline.
    """

    candidate: str
    baseline: str | None
    families: Sequence[str]
    budgets: Sequence[Budget]
    replicates: int
    output: Path
    candidate_options: Sequence[str] = ()
    baseline_options: Sequence[str] = ()


@dataclass(frozen=True, slots=True)
class SideSpec:
    """Hold the engine configuration of one side of a pair.

    Attributes:
        name (str): `candidate` or `baseline`.
        engine (str): `rust` or `cpp`.
        algorithms (tuple[str, ...] | None): Portfolio override, or None for the configured one.
        options (ConfigOverrides): `config.toml` overrides of this side.
        params_dir (Path | None): Parameter directory with the overrides, or None for defaults.
    """

    name: str
    engine: str
    algorithms: tuple[str, ...] | None
    options: ConfigOverrides
    params_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class TrialSpec:
    """Hold the instance, budget and seeding of one trial.

    Attributes:
        instance (GateInstance): Instance to solve.
        budget (Budget): Search budget.
        replicate (int): Replicate number, also used to alternate the run order.
        seed (int): Solver seed shared by both sides of the pair.
    """

    instance: GateInstance
    budget: Budget
    replicate: int
    seed: int


class GateInstances:
    """Select and build the gate instances for one split.

    Responsibilities:
    - Draw disjoint calibration and test samples from fixed seeds.
    - Download and verify-by-hash KPLIB inputs and write shifted Rastrigin inputs.

    Attributes:
        cache_dir (Path): Cache root for KPLIB files and generated Rastrigin shifts.
        split (str): `test` or `calibration`.
        count (int): Instances per family.
    """

    def __init__(self, cache_dir: Path, split: str, count: int) -> None:
        """Store the selection options without touching the network or the disk.

        Args:
            cache_dir (Path): Cache root for inputs.
            split (str): `test` or `calibration`.
            count (int): Positive instances per family.

        Raises:
            ValueError: If the split is unknown or the count is not positive.
        """
        if split not in ("test", "calibration"):
            raise ValueError("split must be 'test' or 'calibration'.")
        if count < 1:
            raise ValueError("instances must be positive.")
        self.cache_dir = cache_dir
        self.split = split
        self.count = count

    def knapsack_ids(self) -> list[tuple[str, int]]:
        """Choose KPLIB (type, id) pairs round-robin over the five types.

        Returns:
            list[tuple[str, int]]: `count` pairs; the test split excludes every calibration id.
        """
        per_type = -(-self.count // len(FAMILIES))
        calibration = sorted(random.Random(CALIBRATION_SEED).sample(range(100), CALIBRATION_POOL))
        if self.split == "calibration":
            ids = calibration[:per_type]
        else:
            pool = [item for item in range(100) if item not in calibration]
            ids = sorted(random.Random(TEST_SEED).sample(pool, per_type))
        return [
            (FAMILIES[position % len(FAMILIES)], ids[position // len(FAMILIES)])
            for position in range(self.count)
        ]

    def build(self, family: str) -> Iterator[GateInstance]:
        """Yield the instances of one family lazily, so compiled objectives are released.

        Args:
            family (str): `knapsack`, `rastrigin-100` or `rastrigin-300`.

        Yields:
            GateInstance: One ready instance.

        Raises:
            ValueError: If the family is unknown.
            OSError: If an input cannot be read, written or downloaded.
        """
        if family == "knapsack":
            for kind, item_id in self.knapsack_ids():
                yield self._knapsack(kind, item_id)
        elif family.startswith("rastrigin-") and int(family.split("-")[1]) in RASTRIGIN_DIMENSIONS:
            for index in range(self.count):
                yield self._rastrigin(int(family.split("-")[1]), index)
        else:
            raise ValueError(f"Unknown family {family!r}.")

    def _knapsack(self, kind: str, item_id: int) -> GateInstance:
        """Load one KPLIB instance and its exact optimum.

        Args:
            kind (str): KPLIB type directory.
            item_id (int): Instance number.

        Returns:
            GateInstance: Native Knapsack problem scored by shortfall to the optimum.

        Raises:
            OSError: If the file cannot be downloaded or read.
            ValueError: If the file is oversized, malformed or not 500 items.
        """
        filename = f"s{item_id:03d}.kp"
        path = self.cache_dir / "kplib" / kind / filename
        if not path.exists():
            url = f"{KPLIB_RAW}/{KPLIB_REVISION}/{kind}/n00500/R01000/{filename}"
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(65537)
            if len(data) > 65536:
                raise ValueError(f"Unexpectedly large KPLIB input: {kind}/{filename}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        instance = read_kp_instance(path)
        if instance.n_items != 500:
            raise ValueError(f"KPLIB item-count mismatch for {kind}/{filename}.")
        optimum = exact_profit(instance)
        problem = problems.Knapsack(instance.benefits, instance.weights, instance.capacity)

        def score(keys: Sequence[float]) -> float:
            """Return the penalized profit shortfall of a key vector.

            Args:
                keys (Sequence[float]): Random keys.

            Returns:
                float: Optimum minus profit, plus the infeasibility penalty.
            """
            return float(problem.evaluate(list(keys))) + optimum

        return GateInstance(
            "knapsack",
            f"{kind}/{filename}",
            hashlib.sha256(path.read_bytes()).hexdigest(),
            float(optimum),
            problem,
            score,
        )

    def _rastrigin(self, dimension: int, index: int) -> GateInstance:
        """Write one shifted Rastrigin instance and build its compiled problem.

        Args:
            dimension (int): Key count.
            index (int): Instance number within the family.

        Returns:
            GateInstance: Numba problem with optimum zero.

        Raises:
            OSError: If the instance file cannot be written.
        """
        seed = CALIBRATION_SEED if self.split == "calibration" else TEST_SEED
        generator = random.Random(f"{seed}:{dimension}:{index}")
        shifts = [round(generator.uniform(-2.0, 2.0), 3) for _ in range(dimension)]
        path = self.cache_dir / "gate-rastrigin" / f"{self.split}-{dimension}-{index:02d}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([str(dimension), *(repr(shift) for shift in shifts)]) + "\n")
        size, compiled, _ = rastrigin_objectives(path)

        def score(keys: Sequence[float]) -> float:
            """Return the shifted Rastrigin value of a key vector.

            Args:
                keys (Sequence[float]): Random keys.

            Returns:
                float: Objective value; the optimum is zero.
            """
            return float(compiled(np.asarray(keys, dtype=np.float64)))

        return GateInstance(
            f"rastrigin-{dimension}",
            path.stem,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            0.0,
            Numba(compiled, size),
            score,
        )


def expand_families(families: Sequence[str]) -> tuple[str, ...]:
    """Expand `rastrigin` into one family per configured dimension.

    Args:
        families (Sequence[str]): `knapsack`, `rastrigin` or a sized family such as `rastrigin-100`.

    Returns:
        tuple[str, ...]: Analysis families in order.
    """
    return tuple(
        name
        for family in families
        for name in (
            [f"rastrigin-{dimension}" for dimension in RASTRIGIN_DIMENSIONS]
            if family == "rastrigin"
            else [family]
        )
    )


def parse_budgets(
    text: str, evaluations: int, evals_time_cap: float = EVALUATION_TIME_CAP
) -> list[Budget]:
    """Turn `2,10,evals` into budgets.

    The `evals` budget runs the Rust workers sequentially (`n_jobs=1`) so the pool never
    exchanges elites by timing, and uses a high time cap so `max_evaluations` is the active limit.

    Args:
        text (str): Comma-separated seconds and the token `evals`.
        evaluations (int): Per-algorithm cap used by the `evals` budget.
        evals_time_cap (float): Wall-clock seconds allowed to the `evals` budget.

    Returns:
        list[Budget]: Parsed budgets in order.

    Raises:
        ValueError: If a token is neither a positive number nor `evals`.
    """
    budgets = []
    for token in text.split(","):
        if token == "evals":
            budgets.append(Budget("evals", evals_time_cap, evaluations, EVALUATION_JOBS))
        elif float(token) > 0:
            budgets.append(Budget(f"{float(token):g}s", float(token), None))
        else:
            raise ValueError(f"Invalid budget {token!r}.")
    return budgets
