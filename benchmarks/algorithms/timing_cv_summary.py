"""Summarize timing variance of completed evaluations in utility pilot rows (issue 99).

Objective:
- Report the median per-instance coefficient of variation (CV) of completed evaluations per
  class and timed profile, so repeated timed runs can be judged against the 0.05 bound.

Technical approach:
- Read JSONL rows written by `utility_cli pilot`, keep the timed profiles (`time2`, `time10`).
- Group evaluations by class, profile and instance across repeats; take the population CV per group.
- Report the median and maximum CV per class and profile, and the median rate CV.

Flow (pseudocode):
1. For each input file, read every row.
2. Group evaluations by (class, profile, instance) and rates by (class, instance) for time2.
3. Print the per-class medians and the rate CV median per file.

Important criteria:
- Only instances with at least two repeats enter a CV.
- The CV is population standard deviation over mean; a zero mean yields NaN and is skipped.

Inputs/Outputs:
- Input: one or more JSONL paths produced by `utility_cli pilot`.
- Output: a text table printed to stdout.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def coefficient_of_variation(values: list[float]) -> float:
    """Return population standard deviation over mean, or NaN when the mean is zero.

    Args:
        values (list[float]): Repeated measurements of one instance.

    Returns:
        float: Coefficient of variation.
    """
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else float("nan")


def read_groups(path: Path) -> tuple[dict, dict]:
    """Collect evaluations per instance and profile, and rates per instance for time2.

    Args:
        path (Path): JSONL file with pilot rows.

    Returns:
        tuple[dict, dict]: Evaluations keyed by (class, profile, instance) and rates keyed by
        (class, instance).
    """
    evaluations: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    rates: dict[tuple[str, str], list[float]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row["profile"].startswith("time"):
            continue
        evaluations[(row["class"], row["profile"], row["instance"])].append(
            float(row["evaluations"])
        )
        if row["profile"] == "time2":
            rate = row["evaluations"] / row["search_seconds"]
            rates[(row["class"], row["instance"])].append(rate)
    return evaluations, rates


def summarize(path: Path) -> None:
    """Print per-class median and maximum CV, and the median rate CV, for one pilot file.

    Args:
        path (Path): JSONL file with pilot rows.
    """
    evaluations, rates = read_groups(path)
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (cls, profile, _), values in evaluations.items():
        if len(values) >= 2:
            groups[(cls, profile)].append(coefficient_of_variation(values))
    print(path.name)
    for (cls, profile), cvs in sorted(groups.items()):
        median = statistics.median(cvs)
        print(f"  {cls:11s} {profile:7s} n={len(cvs):2d} median={median:.3f} max={max(cvs):.3f}")
    rate_cvs = [coefficient_of_variation(v) for v in rates.values() if len(v) >= 2]
    print(f"  evals/s (time2) median CV={statistics.median(rate_cvs):.3f}")


def main() -> None:
    """Summarize every pilot file named on the command line."""
    for name in sys.argv[1:]:
        summarize(Path(name))


if __name__ == "__main__":
    main()
