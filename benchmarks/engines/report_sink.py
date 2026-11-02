"""Shared JSON report writing and per-arm column collection for the decoder-form studies.

Objective:
- Give the packing, traveling-salesman and quantile-codec studies one place to write their JSON
  report and to collect one field of every arm on one family and budget, instead of three
  near-identical copies.

Technical approach:
- `ReportSink.write` dumps a report dict to disk, creating parent directories, and prints its
  verdict to standard output.
- `ReportSink.column` walks normalized trial records once and buckets one field per arm.

Flow (pseudocode):
1. Write accepts the whole report dict and an output path.
2. Column accepts the arm names, the trial rows and a family/budget/field filter.

Important criteria:
- Neither function reads or infers the decision rule; they only move and filter data.

Inputs/Outputs:
- Input: a report dict or a list of trial records, plus an output path or a filter.
- Output: a written file and a stdout line, or a dict of per-arm value lists.
"""

from __future__ import annotations

import json
import numpy as np
import sys
from benchmarks.engines.priority_decoder_study import PairedStatistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

StudyShape = tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...], tuple[float, ...]]


class ReportSink:
    """Write a study's JSON report and collect per-arm trial columns.

    Responsibilities:
    - Persist a report dict to disk and echo its verdict.
    - Bucket one normalized field of every arm on one family and budget.
    """

    @staticmethod
    def write(report: dict[str, Any], output: Path) -> None:
        """Write the JSON report and print the verdict.

        Args:
            report (dict[str, Any]): Must hold `summary.verdict`.
            output (Path): Destination file; parent directories are created.
        """
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        sys.stdout.write(json.dumps(report["summary"]["verdict"], indent=2) + "\n")

    @staticmethod
    def column(
        arms: tuple[str, ...],
        rows: list[dict[str, Any]],
        kind: str,
        budget: float,
        field: str,
    ) -> dict[str, list[float]]:
        """Collect one field of every arm on one family and budget in instance order.

        Args:
            arms (tuple[str, ...]): Arm names to bucket, in a fixed order.
            rows (list[dict[str, Any]]): Normalized trial records.
            kind (str): Instance family.
            budget (float): Per-instance allowance in seconds.
            field (str): Field name to collect, for example `gap` or `bound_gap`.

        Returns:
            dict[str, list[float]]: Values per arm.
        """
        columns: dict[str, list[float]] = {arm: [] for arm in arms}
        for row in rows:
            if row["family"] == kind and row["budget"] == budget:
                columns[row["arm"]].append(row[field])
        return columns

    @staticmethod
    def tables(
        shape: StudyShape, rows: list[dict[str, Any]]
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, dict[str, float]]]]]:
        """Build the mean-gap and paired-statistics tables shared by every decoder-form study.

        Args:
            shape (tuple[...]): Instance families, arm names (fixed order), arm pairs to
                compare, and per-instance allowances in seconds.
            rows (list[dict[str, Any]]): Normalized trial records.

        Returns:
            tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, dict[str, float]]]]]:
            Mean gap per family/budget/arm (plus `<arm>|bound`), and paired statistics per
            family, budget label and `arm-vs-arm` pair.
        """
        kinds, arms, pairs, budgets = shape
        means: dict[str, dict[str, float]] = {}
        paired: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        for kind in kinds:
            paired[kind] = {}
            for budget in budgets:
                gaps = ReportSink.column(arms, rows, kind, budget, "gap")
                bound = ReportSink.column(arms, rows, kind, budget, "bound_gap")
                means[f"{kind}@{budget:g}s"] = {arm: float(np.mean(gaps[arm])) for arm in arms} | {
                    f"{arm}|bound": float(np.mean(bound[arm])) for arm in arms
                }
                paired[kind][f"{budget:g}s"] = {
                    f"{a}-vs-{b}": PairedStatistics.compare(gaps[a], gaps[b], 10000, 0)
                    for a, b in pairs
                }
        return means, paired

    @staticmethod
    def summarize_tables(
        shape: StudyShape,
        rows: list[dict[str, Any]],
        decide: Callable[[dict[str, dict[str, dict[str, dict[str, float]]]]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the mean-gap and paired tables and apply the decision rule to them.

        Args:
            shape (tuple[...]): Instance families, arm names (fixed order), arm pairs to
                compare, and per-instance allowances in seconds.
            rows (list[dict[str, Any]]): Normalized trial records.
            decide (Callable[[...], dict[str, Any]]): Decision rule, called on the paired tables.

        Returns:
            dict[str, Any]: `mean_gap`, `paired` and the `verdict` returned by `decide`.
        """
        means, paired = ReportSink.tables(shape, rows)
        return {"mean_gap": means, "paired": paired, "verdict": decide(paired)}


def require_disjoint(spans: tuple[range, ...], message: str) -> None:
    """Raise if any two of the given ranges overlap.

    Args:
        spans (tuple[range, ...]): Ranges to check pairwise.
        message (str): Error message when two ranges overlap.

    Raises:
        ValueError: If any two ranges overlap.
    """
    for index, first in enumerate(spans):
        for second in spans[index + 1 :]:
            if first.start < second.stop and second.start < first.stop:
                raise ValueError(message)
