"""Command line of the tuning bake-off.

Objective:
- Prepare instances, run the missing sessions and print the held-out report.

Technical approach:
- `prepare` generates and calibrates instances, `run` executes missing sessions with a process
  pool and `report` summarizes a results file; each step is repeatable and `run` resumes.

Flow (pseudocode):
1. Parse the sub-command and options.
2. Build `Settings` and the `BakeOff`.
3. Run the requested step.

Important criteria:
- The entry point is guarded for spawned workers.

Inputs/Outputs:
- Input: sub-command and options.
- Output: files in the work directory and a Markdown report on standard output.
"""

from __future__ import annotations

import argparse
from benchmarks.tuning.analysis import Analysis
from benchmarks.tuning.confirm import Confirmation
from benchmarks.tuning.instances import FAMILIES
from benchmarks.tuning.protocol import TUNERS
from benchmarks.tuning.protocol import BakeOff
from benchmarks.tuning.protocol import Settings
from benchmarks.tuning.spaces import SPACE_PROFILES
from benchmarks.tuning.spaces import SPACES
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested step.

    Args:
        argv (Sequence[str] | None): Command-line arguments, or the process arguments.

    Returns:
        int: Exit status, zero on success.

    Raises:
        SystemExit: If the arguments are invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["prepare", "run", "report", "confirm"])
    parser.add_argument("--directory", type=Path, default=Path("tuning-work"))
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--algorithms", nargs="+", default=list(SPACES), choices=list(SPACES))
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    parser.add_argument("--tuners", nargs="+", default=list(TUNERS), choices=list(TUNERS))
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--train", type=int, default=12)
    parser.add_argument("--test", type=int, default=20)
    parser.add_argument("--test-seeds", type=int, default=5)
    parser.add_argument("--evaluations", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--space-profile", choices=list(SPACE_PROFILES), default="original")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--confirm-evaluations", type=int, default=500_000)
    parser.add_argument("--confirm-instances", type=int, default=10)
    parser.add_argument("--confirm-seeds", type=int, default=3)
    parser.add_argument("--confirm-tuner", choices=list(TUNERS), default="racing")
    args = parser.parse_args(argv)
    results = args.results or args.directory / "results.jsonl"
    settings = Settings(
        str(args.directory),
        args.budget,
        args.train,
        args.test,
        args.test_seeds,
        args.evaluations,
        args.workers,
        args.space_profile,
        args.seed_offset,
    )
    if args.step == "confirm":
        output = args.directory / "confirm.jsonl"
        check = Confirmation(results, output, args.directory, args.confirm_tuner)
        count = check.run(
            args.confirm_evaluations, args.confirm_instances, args.confirm_seeds, args.workers
        )
        print(f"Scored {count} configurations.")
        print(check.report(args.confirm_evaluations))
        return 0
    if args.step == "report":
        print(Analysis(results).report())
        return 0
    bake = BakeOff(settings, results)
    bake.done()
    bake.prepare()
    if args.step == "run":
        count = bake.run(args.algorithms, args.families, args.tuners, args.reps)
        print(f"Executed {count} sessions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
