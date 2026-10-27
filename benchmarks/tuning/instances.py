"""Generated problem instances, calibrated to a common scale, and solver runs on them.

Objective:
- Provide training and held-out instances of three problem families and turn one solver run
  into a normalized score that is comparable across instances.

Technical approach:
- Instances are generated from fixed seeds (disjoint ranges for training and held-out roles)
  in the formats the existing benchmark objectives read, so the objectives are the maintained
  Numba ones from `benchmarks.python_problems`.
- Calibration fixes, per instance, a reference best cost (the best of default-parameter runs of
  every tuned algorithm with a larger evaluation budget) and an anchor, the median cost of the same
  runs stopped after a few hundred evaluations. The score of a run is
  `(cost - best) / (anchor - best)`: 0 matches the reference, 1 matches a barely started search,
  and negative values beat the reference. Random keys are not used as the anchor because the
  infeasibility penalty of Knapsack makes their mean meaningless.
- Runs use the Rust engine with a single worker and an evaluation limit, so the work done never
  depends on machine speed.

Flow (pseudocode):
1. Generate the instance files of a family and role.
2. Calibrate each instance once and cache the reference in a JSON file.
3. Score a run of (algorithm, configuration, seed) as a normalized gap.

Important criteria:
- Calibration uses default parameters only and fixed seeds; a tuner never sees held-out files.
- The compiled objective of an instance is built once per process and reused.

Inputs/Outputs:
- Input: a directory, a family name, a role and a count.
- Output: `Instance` objects whose `score` returns a normalized gap for a solver run.
"""

from __future__ import annotations

import json
import numpy as np
from benchmarks.python_problems import OBJECTIVES
from benchmarks.tuning.spaces import SPACES
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from rko_solver import RKO
from rko_solver import Numba

FAMILIES = ("kp", "tsp", "rastrigin")
ROLE_SEEDS = {"train": 1000, "test": 2000}
CALIBRATION_EVALUATIONS = 150_000
CALIBRATION_SEEDS = 6
ANCHOR_EVALUATIONS = 500


@dataclass(frozen=True)
class Instance:
    """One generated problem instance.

    Responsibilities:
    - Hold the identity and file of an instance and score solver runs on it.

    Attributes:
        family (str): `kp`, `tsp` or `rastrigin`.
        role (str): `train` or `test`.
        index (int): Position inside the role.
        path (Path): Instance file.
        best (float): Reference best cost found at calibration.
        anchor (float): Median cost of barely started default runs.
    """

    family: str
    role: str
    index: int
    path: Path
    best: float = 0.0
    anchor: float = 1.0

    @property
    def name(self) -> str:
        """Return a stable identifier.

        Returns:
            str: `family-role-index`.
        """
        return f"{self.family}-{self.role}-{self.index}"

    def normalize(self, cost: float) -> float:
        """Convert a raw cost into the common scale.

        Args:
            cost (float): Objective value of a run.

        Returns:
            float: 0 at the reference best, 1 at the anchor.
        """
        return (cost - self.best) / (self.anchor - self.best)


class InstanceBank:
    """Generate, calibrate and run instances.

    Responsibilities:
    - Write instance files, compile their objectives once per process and run the solver.
    - Calibrate instances and persist the calibration, one file per family.

    Attributes:
        directory (Path): Folder with instance files and `calibration-<family>.json`.
        size (dict[str, int]): Problem size per family.
        evaluations (int): Evaluation limit of every scored run.
    """

    SIZES = {"kp": 300, "tsp": 60, "rastrigin": 30}

    def __init__(self, directory: Path, evaluations: int = 50_000) -> None:
        """Create a bank rooted at a directory.

        Args:
            directory (Path): Folder for instances and calibration.
            evaluations (int): Evaluation limit of every scored run.
        """
        self.directory = directory
        self.evaluations = evaluations
        self._compiled: dict[str, tuple[int, Callable[[np.ndarray], float]]] = {}
        self._calibration: dict[str, dict[str, float]] = {}
        for stored in sorted(directory.glob("calibration-*.json")):
            self._calibration.update(json.loads(stored.read_text(encoding="utf-8")))

    def generate(self, family: str, role: str, count: int) -> list[Instance]:
        """Write missing instance files of a family and role and return them calibrated if known.

        Args:
            family (str): One of `FAMILIES`.
            role (str): `train` or `test`.
            count (int): Number of instances.

        Returns:
            list[Instance]: Instances in index order.

        Raises:
            ValueError: If the family or role is unknown.
        """
        if family not in FAMILIES or role not in ROLE_SEEDS:
            raise ValueError(f"Unknown family '{family}' or role '{role}'.")
        self.directory.mkdir(parents=True, exist_ok=True)
        found = []
        for index in range(count):
            rng = np.random.default_rng(ROLE_SEEDS[role] + 10 * index + FAMILIES.index(family))
            path = self.directory / f"{family}-{role}-{index}.txt"
            if not path.is_file():
                path.write_text(self._text(family, rng), encoding="utf-8")
            found.append(self._with_calibration(Instance(family, role, index, path)))
        return found

    def _text(self, family: str, rng: np.random.Generator) -> str:
        """Render one instance file.

        Args:
            family (str): Problem family.
            rng (np.random.Generator): Seeded generator.

        Returns:
            str: File content in the format the benchmark objectives read.
        """
        size = self.SIZES[family]
        if family == "kp":
            weights = rng.integers(10, 1001, size)
            profits = np.maximum(1, weights + rng.integers(-100, 101, size))
            rows = "\n".join(f"{p}\t{w}" for p, w in zip(profits, weights, strict=True))
            return f"{size}\t{int(weights.sum() // 2)}\n{rows}\n"
        if family == "tsp":
            points = rng.uniform(0, 1000, (size, 2))
            return f"{size}\n" + "\n".join(f"{x:.3f} {y:.3f}" for x, y in points) + "\n"
        return f"{size}\n" + " ".join(f"{s:.4f}" for s in rng.uniform(-2, 2, size)) + "\n"

    def _with_calibration(self, instance: Instance) -> Instance:
        """Attach the cached calibration to an instance when it exists.

        Args:
            instance (Instance): Instance without calibration.

        Returns:
            Instance: Calibrated copy, or the input when nothing is cached.
        """
        known = self._calibration.get(instance.name)
        if known is None:
            return instance
        return Instance(
            instance.family,
            instance.role,
            instance.index,
            instance.path,
            known["best"],
            known["anchor"],
        )

    def objective(self, instance: Instance) -> tuple[int, Callable[[np.ndarray], float]]:
        """Return the compiled objective of an instance, building it once per process.

        Only the objectives of one family are kept: compiled code is the largest memory cost of a
        worker, and sessions of a family run together.

        Args:
            instance (Instance): Instance to compile.

        Returns:
            tuple[int, Callable[[np.ndarray], float]]: Dimension and the njit objective.
        """
        if instance.name not in self._compiled:
            self._compiled = {
                name: value
                for name, value in self._compiled.items()
                if name.startswith(f"{instance.family}-")
            }
            dimension, compiled, _ = OBJECTIVES[instance.family](instance.path)
            compiled(np.full(dimension, 0.5))
            self._compiled[instance.name] = (dimension, compiled)
        return self._compiled[instance.name]

    def cost(
        self, instance: Instance, algorithm: str, config: dict[str, float], seed: int
    ) -> float:
        """Run the solver once and return the raw cost.

        Args:
            instance (Instance): Instance to solve.
            algorithm (str): Tuned algorithm.
            config (dict[str, float]): Offline parameters.
            seed (int): Run seed.

        Returns:
            float: Best objective value after the evaluation limit.

        Raises:
            ValueError: If the solver rejects the configuration.
        """
        dimension, compiled = self.objective(instance)
        result = RKO(
            Numba(compiled, dimension),
            engine="rust",
            algorithms=[algorithm],
            adaptation="fixed",
            hyperparameters={algorithm: config},
            random_state=seed,
            max_evaluations=self.evaluations,
            n_jobs=1,
        ).solve(3600.0)
        return float(result.cost)

    def score(
        self, instance: Instance, algorithm: str, config: dict[str, float], seed: int
    ) -> float:
        """Run the solver once and return the normalized gap.

        Args:
            instance (Instance): Calibrated instance.
            algorithm (str): Tuned algorithm.
            config (dict[str, float]): Offline parameters.
            seed (int): Run seed.

        Returns:
            float: Normalized gap of the run.
        """
        return instance.normalize(self.cost(instance, algorithm, config, seed))

    def calibrate(self, instances: list[Instance]) -> list[Instance]:
        """Calibrate instances once and persist the result.

        Args:
            instances (list[Instance]): Instances, possibly already calibrated.

        Returns:
            list[Instance]: Calibrated instances.
        """
        calibrated = []
        for instance in instances:
            if instance.name not in self._calibration:
                self._calibration[instance.name] = self._measure(instance)
                mine = {k: v for k, v in self._calibration.items() if k.startswith(instance.family)}
                path = self.directory / f"calibration-{instance.family}.json"
                path.write_text(json.dumps(mine, indent=1, sort_keys=True), encoding="utf-8")
            calibrated.append(self._with_calibration(instance))
        return calibrated

    def _measure(self, instance: Instance) -> dict[str, float]:
        """Measure the reference best and the anchor of one instance.

        Args:
            instance (Instance): Instance to measure.

        Returns:
            dict[str, float]: `best` and `anchor`.
        """
        previous = self.evaluations
        runs = [
            (name, space.default(), 7000 + seed)
            for name, space in SPACES.items()
            for seed in range(CALIBRATION_SEEDS)
        ]
        try:
            self.evaluations = CALIBRATION_EVALUATIONS
            best = min(self.cost(instance, name, config, seed) for name, config, seed in runs)
            self.evaluations = ANCHOR_EVALUATIONS
            anchor = float(np.median([self.cost(instance, *run) for run in runs]))
        finally:
            self.evaluations = previous
        return {"best": float(best), "anchor": max(anchor, best + 1e-9)}
