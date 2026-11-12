"""Diagnose the simulated-annealing start temperature on KPLIB Knapsack inputs.

Objective:
- Record the estimated start temperature, cutoff, temperature levels and calibration sample for
  each KPLIB input, with and without the sampled estimator, at a fixed evaluation budget.

Technical approach:
- Draw one pilot-role input per KPLIB family and size through `UtilityInputs`, which is the
  diagnosis cohort (not a verdict cohort).
- Run the repository `rko_bench` binary twice per input on the same seed and parameters, once with
  the packaged parameters and once with `--sa-t0-auto`, and keep the JSON summary of each run.

Flow (pseudocode):
1. Locate the release `rko_bench` binary and the KPLIB cache.
2. For each size and family, draw one pilot input and resolve its file.
3. Run both arms and append one JSON Lines row per run.

Important criteria:
- Both arms use the same seed, evaluation budget and SA parameter vector.
- Rows are appended as each run finishes, so an interrupted campaign keeps its completed evidence.

Inputs/Outputs:
- Input: repository root, output path and optional `rko_bench` binary path.
- Output: a JSON Lines file with one object per run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from benchmarks.algorithms.utility_inputs import UtilityInputs
from pathlib import Path

PACKAGED_SA = (
    "1027.0,340.3863933617792,0.8601360548373771,0.0035204893377555175,0.008242788398191462"
)
EVALUATIONS = 100000
ARMS = (("baseline", []), ("estimator", ["--sa-t0-auto"]))


def run_arm(binary: Path, path: Path, flags: list[str]) -> dict[str, object]:
    """Run one SA search and return its JSON summary.

    Args:
        binary (Path): Release `rko_bench` executable.
        path (Path): KPLIB input file.
        flags (list[str]): Extra arguments selecting the arm.

    Returns:
        dict[str, object]: Parsed JSON summary of the last output line.
    """
    command = [
        str(binary),
        str(path),
        "600",
        "--problem",
        "kp",
        "--algo",
        "SA",
        "--fixed",
        "--threads",
        "1",
        "--seed",
        "1",
        "--sa-parameters",
        PACKAGED_SA,
        "--evaluations",
        str(EVALUATIONS),
        *flags,
    ]
    done = subprocess.run(command, capture_output=True, text=True, timeout=900, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


def main(argv: list[str]) -> None:
    """Run the diagnosis cohort and append rows to the output file.

    Args:
        argv (list[str]): Repository root, output path and optional binary path.
    """
    repo = Path(argv[0])
    output = Path(argv[1])
    binary = Path(argv[2]) if len(argv) > 2 else repo / "target" / "release" / "rko_bench"
    inputs = UtilityInputs(Path.home() / ".cache" / "rko-cpp-python-knapsack")
    with output.open("a", encoding="utf-8") as sink:
        for size in (500, 1000):
            for kind, item in inputs.draw(size, 1, "pilot"):
                path = inputs.kplib_path(kind, size, item)
                for arm, flags in ARMS:
                    summary = run_arm(binary, path, flags)
                    row = {
                        "size": size,
                        "kind": kind,
                        "item": item,
                        "arm": arm,
                        "cost": summary["cost"],
                        "evaluations": summary["evaluations"],
                        "annealing": summary["annealing"],
                    }
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
