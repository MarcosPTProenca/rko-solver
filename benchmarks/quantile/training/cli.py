"""Command-line entry point for the bounded, processor-only training pilot.

Objective:
- Translate explicit command-line arguments into one locked, resource-capped training run.

Technical approach:
- Parse arguments, apply hard process limits, then delegate to `TrainingRun.run`.

Flow (pseudocode):
1. Parse the required family, size, step and processor-second arguments.
2. Reject a CUDA build and fix single-threaded, address- and CPU-time-limited execution.
3. Run the bounded training lifecycle and print its compact status summary.

Important criteria:
- No behavior beyond argument parsing, resource limits and process exit status.

Inputs/Outputs:
- Input: command-line arguments naming family, size, steps, seed and output directory.
- Output: a status line on standard output and a zero or nonzero process exit code.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import torch
from benchmarks.quantile.training.run import TrainingRun
from benchmarks.quantile.training.run import TrainingSettings
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run the explicitly isolated Linux processor-only training command.

    Args:
        argv (list[str] | None): Explicit argument vector; omit to parse `sys.argv`.

    Returns:
        int: Zero for a completed development run, two for resource exhaustion.

    Raises:
        ValueError: If settings, package backend or model validation are invalid.
        OSError: If resources or output paths cannot be configured.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=("tour", "cover"), required=True)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--cpu-seconds", type=float, default=1500.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    settings = TrainingSettings(args.family, args.size, args.steps, args.cpu_seconds, args.seed)
    if torch.version.cuda is not None:
        raise ValueError("This pilot requires the locked processor-only framework build.")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    limit = math.ceil(settings.cpu_seconds)
    resource.setrlimit(resource.RLIMIT_CPU, (limit + 1, limit + 5))
    report = TrainingRun(settings, args.output).run()
    print(
        json.dumps(
            {key: report[key] for key in ("status", "advance", "processor_seconds", "comparisons")}
        ),
        flush=True,
    )
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
