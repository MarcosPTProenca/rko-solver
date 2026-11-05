"""Repository-only, offline simulated annealing temperature screen.

Objective:
- Test an unshipped temperature/cooling-floor composite without exposing a Python option.

Technical approach:
- Reserve fresh KPLIB IDs and shifted Rastrigin inputs before comparing one native binary.
- Score returned keys independently and reuse the existing non-inferiority report.

Flow (pseudocode):
1. Freeze a disjoint input cohort and validate the native benchmark executable.
2. Run paired offline SA trials under identical evaluation budgets.
3. Record provenance and report the pre-registered non-inferiority outcome.

Important criteria:
- This is a screen of a composite, not a public feature or an isolated temperature effect.
- No time-limited, failed or incorrectly budgeted trial may enter the quality comparison.

Inputs/Outputs:
- Input: a repository-only native benchmark and a fresh output path.
- Output: paired trial rows and the existing gate report.
"""

import argparse
import hashlib
import json
import math
import numpy as np
import platform
import random
import subprocess
from benchmarks.engines.cpp_python_knapsack import FAMILIES
from benchmarks.engines.gate import DEFAULT_CACHE
from benchmarks.engines.gate import Gate
from benchmarks.engines.gate_instances import GateInstance
from benchmarks.engines.gate_instances import GateInstances
from benchmarks.engines.gate_support import GateReport
from benchmarks.problems.numeric import rastrigin_objectives
from collections.abc import Iterator
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SA_SCREEN_SEED = 20261122
PROGRESS_CONFIRMATION_SEED = 20261121
EVALUATIONS = 100_000


class ScreenInputs(GateInstances):
    """Build only the frozen offline screen inputs.

    Responsibilities:
    - Exclude the calibration, historical test and progress-confirmation cohorts.
    - Keep generated shifts in a separate namespace.

    Attributes:
        cache_dir (Path): Reused KPLIB cache and fresh shift directory.
        count (int): Twenty instances per family in this frozen screen.
    """

    def __init__(self, cache_dir: Path, count: int = 20) -> None:
        """Bind the screen to its cache and instance count.

        Args:
            cache_dir (Path): Input cache root.
            count (int): Instances to draw; the frozen screen used 20 and a larger value
                exposes the whole reserved pool.
        """
        super().__init__(cache_dir, "test", count)

    def knapsack_ids(self) -> list[tuple[str, int]]:
        """Select twenty fresh, repeatable KPLIB inputs.

        Returns:
            list[tuple[str, int]]: Type and identifier pairs outside all reserved cohorts.
        """
        calibration = {
            number for _, number in GateInstances(self.cache_dir, "calibration", 100).knapsack_ids()
        }
        historical = {
            number for _, number in GateInstances(self.cache_dir, "test", 100).knapsack_ids()
        }
        pool = sorted(set(range(100)) - calibration - historical)
        progress = random.Random(PROGRESS_CONFIRMATION_SEED).sample(pool, 20)
        pool = sorted(set(pool) - set(progress))
        fresh = sorted(random.Random(SA_SCREEN_SEED).sample(pool, 20))
        return [
            (FAMILIES[position % len(FAMILIES)], fresh[position // len(FAMILIES)])
            for position in range(self.count)
        ]

    def rastrigin_path(self, dimension: int, index: int) -> Path:
        """Write one deterministic shifted input without reusing an earlier namespace.

        Args:
            dimension (int): Supported key count, 100 or 300.
            index (int): Input number in the frozen cohort.

        Returns:
            Path: Generated input file.

        Raises:
            ValueError: On an unsupported dimension or index.
            OSError: If the input file cannot be written.
        """
        if dimension not in (100, 300) or not 0 <= index < self.count:
            raise ValueError("screen requires dimensions 100/300 and indices 0..19")
        generator = random.Random(f"{SA_SCREEN_SEED}:{dimension}:{index}")
        shifts = [round(generator.uniform(-2.0, 2.0), 3) for _ in range(dimension)]
        path = self.cache_dir / "sa-screen" / f"sa-screen-{dimension}-{index:02d}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([str(dimension), *(repr(value) for value in shifts)]) + "\n")
        return path


class Screen:
    """Run and validate the fixed paired native screen.

    Responsibilities:
    - Refuse native output that does not match the frozen budget.
    - Independently score returned keys before recording quality.

    Attributes:
        binary (Path): Native benchmark executable.
        inputs (ScreenInputs): Frozen study inputs.
        output (Path): New paired-trial output path.
    """

    def __init__(self, binary: Path, inputs: ScreenInputs, output: Path) -> None:
        """Bind the screen without running or overwriting any trial.

        Args:
            binary (Path): Native benchmark executable.
            inputs (ScreenInputs): Frozen study inputs.
            output (Path): New paired-trial output path.
        """
        self.binary = binary.resolve()
        self.inputs = inputs
        self.output = output

    def run(self) -> int:
        """Run the frozen pairs once and return the non-inferiority gate exit code.

        Returns:
            int: Zero for all-family non-inferiority; one for a rejected candidate.

        Raises:
            FileExistsError: If the output already exists.
            RuntimeError: If any native trial fails validation.
            OSError: If the binary, inputs or output cannot be accessed.
        """
        self.output.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        with self.output.open("x") as output:
            commit, dirty = Gate.git_state()
            provenance = {
                "git_commit": commit,
                "git_dirty": dirty,
                "host": platform.platform(),
                "native_path": str(self.binary),
                "native_sha256": hashlib.sha256(self.binary.read_bytes()).hexdigest(),
                "binary_kind": "rko_bench",
                "split": "sa-screen",
                "cohort_seed": SA_SCREEN_SEED,
            }
            for case in self.cases():
                for replicate in range(5):
                    sides = (False, True) if replicate % 2 == 0 else (True, False)
                    for candidate in sides:
                        row = {**provenance, **self._trial(case, replicate, candidate)}
                        output.write(json.dumps(row, allow_nan=False) + "\n")
                        output.flush()
                        rows.append(row)
        if len(rows) != 600:
            raise RuntimeError("frozen screen requires exactly 600 trial rows")
        cells = GateReport(0.02, floors={"knapsack": 0.0, "rastrigin": 0.0}).analyze(rows)
        print(GateReport.render(cells))
        return int(any(cell["verdict"] != "pass" for cell in cells))

    def cases(self) -> Iterator[tuple[GateInstance, Path]]:
        """Yield the frozen inputs and their native-readable file paths.

        Yields:
            tuple[GateInstance, Path]: Independent scorer and native input file.

        Raises:
            OSError: If an input cannot be downloaded or written.
            ValueError: If a cached input is malformed.
        """
        for instance in self.inputs.build("knapsack"):
            yield instance, self.inputs.cache_dir / "kplib" / instance.name
        for dimension in (100, 300):
            for index in range(self.inputs.count):
                yield self._rastrigin_case(dimension, index)

    def _rastrigin_case(self, dimension: int, index: int) -> tuple[GateInstance, Path]:
        """Build a native file and independent NumPy scorer for one shifted Rastrigin input.

        Args:
            dimension (int): Key count.
            index (int): Frozen instance number.

        Returns:
            tuple[GateInstance, Path]: Scorer and native-readable input.

        Raises:
            OSError: If the generated file cannot be accessed.
            ValueError: If the input coordinates are unsupported.
        """
        path = self.inputs.rastrigin_path(dimension, index)
        _, _, blocked = rastrigin_objectives(path)

        def score(keys: Sequence[float]) -> float:
            """Independently score one vector using the existing block objective.

            Args:
                keys (Sequence[float]): Native-returned best keys.

            Returns:
                float: Shifted Rastrigin value.
            """
            return float(blocked(np.asarray([keys], dtype=np.float64))[0])

        instance = GateInstance(
            f"rastrigin-{dimension}",
            path.stem,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            0.0,
            None,
            score,
        )
        return instance, path

    def _trial(
        self, case: tuple[GateInstance, Path], replicate: int, candidate: bool
    ) -> dict[str, Any]:
        """Run one scalar offline SA trial and verify its declared experimental mode.

        Args:
            case (tuple[GateInstance, Path]): Scorer and native input.
            replicate (int): Paired solver-seed index.
            candidate (bool): Whether to enable the temperature/cutoff composite.

        Returns:
            dict[str, Any]: Validated measurements and paired-run configuration.

        Raises:
            RuntimeError: If the process fails or reports a different experimental mode.
            ValueError: If the objective or budget metadata is invalid.
            subprocess.TimeoutExpired: If the safety cap plus grace period expires.
        """
        instance, path = case
        seed = 20261201 + replicate
        kind = "kp" if instance.family == "knapsack" else "rastrigin"
        command = [
            str(self.binary),
            str(path),
            "600",
            "--problem",
            kind,
            "--algo",
            "SA",
            "--fixed",
            "--threads",
            "1",
            "--seed",
            str(seed),
            "--evaluations",
            str(EVALUATIONS),
        ]
        if candidate:
            command.append("--sa-t0-auto")
        process = subprocess.run(command, capture_output=True, text=True, timeout=630, check=False)
        if process.returncode:
            raise RuntimeError(
                f"native trial failed ({process.returncode}): {process.stderr[-2000:]}"
            )
        measurement = self.parse_result(process.stdout, instance)
        if measurement["sa_t0_auto"] is not candidate:
            raise RuntimeError("native result reports a different temperature mode")
        return {
            **measurement,
            "family": instance.family,
            "instance": instance.name,
            "instance_sha256": instance.sha256,
            "reference_value": instance.reference,
            "budget": "evals",
            "max_evaluations": EVALUATIONS,
            "n_jobs": 1,
            "time_cap": 600,
            "replicate": replicate,
            "seed": seed,
            "side": "candidate" if candidate else "baseline",
            "status": "ok",
            "spec": "rust/SA/offline",
            "options": {"control": 0, "progress_schedule": False, "sa_t0_auto": candidate},
        }

    @staticmethod
    def _native_output(stdout: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Parse native vectors and refuse a different budget or adaptive mode.

        Args:
            stdout (str): Complete native benchmark output.

        Returns:
            tuple[dict[str, Any], dict[str, Any]]: Keys/objective and run summary.

        Raises:
            ValueError: If the JSON or budget metadata is invalid.
            StopIteration: If the result-vector line is missing.
        """
        lines = stdout.splitlines()
        result = json.loads(
            next(
                line.removeprefix("RKO_BENCH_RESULT ")
                for line in lines
                if line.startswith("RKO_BENCH_RESULT ")
            )
        )
        summary = json.loads(lines[-1])
        expected = {
            "evaluations": EVALUATIONS,
            "stops": ["max_evaluations"],
            "workers": 1,
            "adaptive": False,
        }
        if any(summary[key] != value for key, value in expected.items()):
            raise ValueError("native result violates the frozen offline evaluation budget")
        return result, summary

    @staticmethod
    def parse_result(stdout: str, instance: GateInstance) -> dict[str, Any]:
        """Validate native output and return independently verified measurements.

        Args:
            stdout (str): Native benchmark output.
            instance (GateInstance): Independent scorer and known reference.

        Returns:
            dict[str, Any]: Verified gap, budget and stop metadata.

        Raises:
            ValueError: If output, budget or independently scored cost is invalid.
        """
        result, summary = Screen._native_output(stdout)
        gap = instance.score(result["keys"])
        cost = gap - instance.reference
        if not math.isclose(cost, result["objective"], rel_tol=1e-10, abs_tol=1e-7):
            raise ValueError("native objective does not match independent scoring")
        if not math.isfinite(gap):
            raise ValueError("nonfinite independent gap")
        normalized = 100 * gap / instance.reference if instance.reference else gap
        return {
            "gap": gap,
            "normalized_gap": normalized,
            "evaluations": summary["evaluations"],
            "stop_reason": summary["stops"][0],
            "elapsed_seconds": summary["elapsed"],
            "sa_t0_auto": summary["sa_t0_auto"],
        }


def main() -> int:
    """Run the fixed, repository-only offline temperature screen.

    Returns:
        int: Zero for all-family non-inferiority, one for a rejected candidate.

    Raises:
        OSError: If the binary, input cache or output cannot be accessed.
        RuntimeError: If any native trial fails validation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    return Screen(args.binary, ScreenInputs(args.cache_dir), args.output).run()


if __name__ == "__main__":
    raise SystemExit(main())
