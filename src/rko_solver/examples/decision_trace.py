"""Demonstrate opt-in original-decoder decisions with the existing scheduling callbacks.

Objective:
- Show bounded authored events without final-key replay or objective reevaluation.

Technical approach:
- Bind the unchanged scheduling callbacks and enable TraceConfig on ordinary RKO.
- Print immutable original-observation evidence; optionally render an offline report.

Flow (pseudocode):
1. Construct one small in-memory scheduling instance and bind its callbacks.
2. Run a sequential portfolio with bounded original-decoder recording.
3. Inspect the selected original observation without calling decode/cost again.
4. Optionally write its existing numerical/report snapshot after optimization.

Important criteria:
- Explicit events are partial author observations, not complete interpreter traces.
- HTML requires the independent reports extra; core recording uses standard-library code.

Inputs/Outputs:
- Input: optional positive time limit and offline HTML output path.
- Output: bounded evidence JSON and optional self-contained HTML report.
"""

import argparse
import json
from ..reporting import OptimizationReport
from ..solver import RKO
from ..tracing import TraceConfig
from .instance_pipeline import SchedulingInstance
from .instance_pipeline import SchedulingProblem
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    """Run the existing instrumented adapter without modifying callback signatures.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments; None reads process arguments.

    Raises:
        ValueError: If solve duration or optional output settings are invalid.
        ImportError: If HTML is requested without the reports extra.
        OSError: If the optional report cannot be written.
    """
    parser = argparse.ArgumentParser(description="Inspect original scheduling decoder decisions.")
    parser.add_argument("--time-limit", type=float, default=0.1)
    parser.add_argument("--html", type=Path)
    args = parser.parse_args(argv)
    problem = SchedulingProblem(SchedulingInstance("decision-demo", (3, 2, 5), (3, 6, 9)))
    result = RKO(
        problem.decode,
        problem.cost,
        problem.n_variables,
        algorithms=["SA", "PSO"],
        execution="sequential",
        adaptation="fixed",
        random_state=42,
        trace=TraceConfig(),
    ).solve(time_limit=args.time_limit)
    assert result.trace is not None
    print(json.dumps(result.trace.to_dict(), indent=2))
    if args.html is not None:
        OptimizationReport(result).write_html(args.html)


if __name__ == "__main__":
    main()
