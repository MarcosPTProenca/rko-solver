"""Summarize public-instance JSONL rows into one table per family (issue 156).

Objective:
- Turn the raw rows of `benchmarks.public.study` into the per-family medians the README reports.

Technical approach:
- Group rows by (family, arm, seconds). For each group report the number of rows, feasible rows,
  median and best gap over feasible rows with a reference, and median evaluations per second.
- Gaps are positive when worse than the reference; a row without reference contributes no gap.

Flow (pseudocode):
1. Read every JSONL row; skip nothing silently.
2. Group rows and compute the statistics per group.
3. Print one table per family.

Important criteria:
- Feasible-only gaps: an infeasible row is counted in `infeasible`, never as a gap.
- Evaluations per second use the wall time of the row.

Inputs/Outputs:
- Input: JSONL path.
- Output: text tables on stdout.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize(rows: list[dict[str, Any]]) -> dict[tuple[str, str, float], dict[str, Any]]:
    """Return the statistics of every (family, arm, seconds) group.

    Args:
        rows (list[dict[str, object]]): Parsed JSONL rows.

    Returns:
        dict[tuple[str, str, float], dict[str, object]]: Counts, gap medians and evaluation rates.
    """
    provenance = {
        (row.get("host"), row.get("commit"), row.get("benchmark_sha256"), row.get("schema_version"))
        for row in rows
    }
    if len(provenance) > 1:
        raise ValueError("Mixed provenance: report one machine and benchmark revision at a time.")
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["family"]), str(row["arm"]), float(row["seconds"]))].append(row)
    return {key: _group(values) for key, values in sorted(groups.items())}


def _median(values: list[float]) -> float | None:
    """Return the median, or no statistic for an empty group.

    Args:
        values (list[float]): Observations to summarize.

    Returns:
        float | None: Median when observations exist.
    """
    return statistics.median(values) if values else None


def _group(values: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute feasible-only gaps and observed evaluation rates.

    Args:
        values (list[dict[str, Any]]): Rows of one family, arm and budget.

    Returns:
        dict[str, Any]: Counts and descriptive statistics; absent references stay null.
    """
    feasible = [row for row in values if row["feasible"]]
    gaps = [float(row["gap"]) for row in feasible if row["gap"] is not None]
    rates = [
        float(row["evaluations"]) / float(row["wall_seconds"])
        for row in values
        if float(row["wall_seconds"]) > 0
    ]
    return {
        "rows": len(values),
        "feasible": len(feasible),
        "infeasible": len(values) - len(feasible),
        "gap_rows": len(gaps),
        "median_gap": _median(gaps),
        "best_gap": min(gaps, default=None),
        "median_evals_per_second": _median(rates),
    }


def render(stats: dict[tuple[str, str, float], dict[str, Any]]) -> str:
    """Render the statistics as one Markdown table per family.

    Args:
        stats (dict[tuple[str, str, float], dict[str, object]]): Output of `summarize`.

    Returns:
        str: Markdown text.
    """
    lines: list[str] = []
    families = sorted({family for family, _, _ in stats})
    for family in families:
        lines.append(f"### {family}")
        lines.append("")
        lines.append("| Arm | Seconds | Rows | Feasible | Median gap | Best gap | Median evals/s |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for (fam, arm, seconds), value in stats.items():
            if fam != family:
                continue
            lines.append(
                f"| {arm} | {seconds:g} | {value['rows']} | {value['feasible']} | "
                f"{_fmt(value['median_gap'])} | {_fmt(value['best_gap'])} | "
                f"{_fmt(value['median_evals_per_second'], 0)} |"
            )
        lines.append("")
    return "\n".join(lines)


def _fmt(value: float | int | None, digits: int = 4) -> str:
    """Format an observed number or a missing-value marker.

    Args:
        value (float | int | None): Statistic to render.
        digits (int): Decimal places.

    Returns:
        str: Fixed-precision number, or a dash for missing observations.
    """
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def main() -> None:
    """Read a JSONL file and print the family tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path)
    arguments = parser.parse_args()
    parsed = [json.loads(line) for line in arguments.rows.read_text(encoding="utf-8").splitlines()]
    print(render(summarize(parsed)))


if __name__ == "__main__":
    main()
