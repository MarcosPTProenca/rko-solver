"""Command-line entry point for the F5b Tour-only confirmation.

Objective:
- Translate command-line settings into one bounded, output-isolated confirmation run.

Technical approach:
- Load the frozen tour checkpoint, build a `Confirmation` and run its complete lifecycle.

Flow (pseudocode):
1. Parse the checkpoint path and the output directory.
2. Load the frozen scorer and construct the confirmation.
3. Run it and print the compact advance decision.

Important criteria:
- No behavior beyond argument parsing and delegating to `Confirmation.run`.

Inputs/Outputs:
- Input: command-line checkpoint path and an output directory.
- Output: a status line on standard output and a zero or nonzero exit code.
"""

from __future__ import annotations

import argparse
import json
import sys
from benchmarks.quantile.confirm import Confirmation
from benchmarks.quantile.model import FrozenScorer
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run the confirmation and report its advance decision.

    Args:
        argv (list[str] | None): Explicit argument vector; omit to parse `sys.argv`.

    Returns:
        int: Zero if the confirmation completed and advanced, two otherwise.

    Raises:
        ValueError: If the checkpoint is not a tour scorer.
        OSError: If the checkpoint or output cannot be read or persisted.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tour-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = Confirmation(args.output, FrozenScorer.load(args.tour_model)).run()
    print(json.dumps({"advance": summary["advance"], "noise": summary["noise"]}), flush=True)
    return 0 if summary["advance"] else 2


if __name__ == "__main__":
    sys.exit(main())
