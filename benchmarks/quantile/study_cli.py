"""Command-line entry point for the F5 frozen-model development screen.

Objective:
- Translate command-line settings into one bounded, output-isolated F5 screen run.

Technical approach:
- Load two frozen checkpoints, build a `Screen` and run its complete lifecycle.

Flow (pseudocode):
1. Parse per-family checkpoint paths and the output directory.
2. Load both frozen scorers and construct the screen.
3. Run the screen and print its compact advance decision.

Important criteria:
- No behavior beyond argument parsing and delegating to `Screen.run`.

Inputs/Outputs:
- Input: command-line checkpoint paths and an output directory.
- Output: a status line on standard output and a zero or nonzero exit code.
"""

from __future__ import annotations

import argparse
import json
import sys
from benchmarks.quantile.model import FrozenScorer
from benchmarks.quantile.study import FAMILIES
from benchmarks.quantile.study import Screen
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run the F5 screen for both families and report its advance decision.

    Args:
        argv (list[str] | None): Explicit argument vector; omit to parse `sys.argv`.

    Returns:
        int: Zero if the screen completed and advanced, two otherwise.

    Raises:
        ValueError: If a checkpoint's family does not match its declared argument.
        OSError: If a checkpoint or output cannot be read or persisted.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for family in FAMILIES:
        parser.add_argument(f"--{family}-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    checkpoints = {
        family: FrozenScorer.load(getattr(args, f"{family}_model")) for family in FAMILIES
    }
    summary = Screen(args.output, checkpoints).run()
    print(json.dumps({"advance": summary["advance"], "wins": summary["wins"]}), flush=True)
    return 0 if summary["advance"] else 2


if __name__ == "__main__":
    sys.exit(main())
