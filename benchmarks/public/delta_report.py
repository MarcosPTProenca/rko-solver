"""Compare swap-delta arms with their plain counterparts from public-instance rows (issue 157).

Objective:
- Quantify whether an exact swap delta changes solution quality, evaluation rate and paired
  outcomes, from the JSONL rows written by `benchmarks.public.study`.

Technical approach:
- Group rows by arm and budget for medians of gap and evaluation rate.
- Pair each `+delta` arm with its plain arm on the same instance, algorithm and budget, using the
  median objective over seeds; count wins, losses and ties.
- Report a two-sided exact sign test over the non-tied pairs, with no multiplicity correction.

Flow (pseudocode):
1. Read JSONL rows and reject mixed provenance.
2. Summarize each arm and budget.
3. For each algorithm and budget, count paired wins, losses and ties and compute the sign test.
4. Print Markdown tables.

Important criteria:
- Evaluation rate uses observed `wall_seconds`, not the requested budget.
- Sign-test p-values are descriptive, one per comparison; no multiplicity correction is applied.

Inputs/Outputs:
- Input: path to a JSONL file written by `benchmarks.public.study`.
- Output: Markdown tables on stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize_arms(rows: list[dict[str, Any]]) -> dict[tuple[str, float], dict[str, float]]:
    """Summarize gap and evaluation rate for every arm and budget.

    Args:
        rows (list[dict[str, Any]]): Parsed JSONL rows with a single provenance.

    Returns:
        dict[tuple[str, float], dict[str, float]]: Median gap, best gap, median evals per second,
        total evaluations and total delta evaluations for each (arm, budget).
    """
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["arm"]), float(row["seconds"]))].append(row)
    summary: dict[tuple[str, float], dict[str, float]] = {}
    for key, values in sorted(groups.items()):
        gaps = [float(row["gap"]) for row in values]
        rates = [float(row["evaluations"]) / float(row["wall_seconds"]) for row in values]
        summary[key] = {
            "median_gap": statistics.median(gaps),
            "best_gap": min(gaps),
            "median_evals_per_second": statistics.median(rates),
            "evaluations": float(sum(row["evaluations"] for row in values)),
            "delta_evaluations": float(sum(row["delta_evaluations"] for row in values)),
        }
    return summary


def sign_test_p(wins: int, losses: int) -> float:
    """Return the two-sided exact sign-test p-value for paired wins and losses.

    Args:
        wins (int): Pairs where the delta arm is better.
        losses (int): Pairs where the delta arm is worse.

    Returns:
        float: Two-sided p-value in [0, 1]; 1.0 when there are no untied pairs.
    """
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, count) for count in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / 2**trials)


def _tally(outcomes: list[str]) -> tuple[int, int, int]:
    """Count wins, losses and ties among pair outcomes.

    Args:
        outcomes (list[str]): "win", "loss" or "tie" for each instance.

    Returns:
        tuple[int, int, int]: Wins, losses and ties.
    """
    return outcomes.count("win"), outcomes.count("loss"), outcomes.count("tie")


def _objectives(
    rows: list[dict[str, Any]], seconds: float, arms: tuple[str, str]
) -> dict[tuple[str, str], list[float]]:
    """Collect objectives over seeds for the given arms at one budget.

    Args:
        rows (list[dict[str, Any]]): Parsed JSONL rows.
        seconds (float): Budget to keep.
        arms (tuple[str, str]): Arm keys to keep.

    Returns:
        dict[tuple[str, str], list[float]]: Objectives keyed by (instance, arm).
    """
    collected: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if float(row["seconds"]) == seconds and str(row["arm"]) in arms:
            collected[(str(row["instance"]), str(row["arm"]))].append(float(row["objective"]))
    return collected


def _matched(
    objectives: dict[tuple[str, str], list[float]], plain_arm: str, delta_arm: str
) -> list[tuple[list[float], list[float]]]:
    """Pair the objectives of both arms for every instance that ran under both.

    Args:
        objectives (dict[tuple[str, str], list[float]]): Objectives keyed by (instance, arm).
        plain_arm (str): Arm key without the delta.
        delta_arm (str): Arm key with the delta.

    Returns:
        list[tuple[list[float], list[float]]]: Plain and delta objectives per shared instance,
        in instance order.
    """
    shared = sorted(
        {instance for instance, arm in objectives if arm == plain_arm}
        & {instance for instance, arm in objectives if arm == delta_arm}
    )
    return [(objectives[(i, plain_arm)], objectives[(i, delta_arm)]) for i in shared]


def _pair_outcome(plain: list[float], delta: list[float]) -> tuple[str, float]:
    """Classify one instance by the median objectives of the two arms.

    Args:
        plain (list[float]): Objectives of the plain arm over seeds.
        delta (list[float]): Objectives of the delta arm over seeds.

    Returns:
        tuple[str, float]: "win", "loss" or "tie" for the delta arm, and its relative difference
        of the median objective, negative when the delta arm is better.
    """
    plain_median = statistics.median(plain)
    delta_median = statistics.median(delta)
    difference = (delta_median - plain_median) / max(1.0, abs(plain_median))
    if delta_median < plain_median:
        return "win", difference
    if delta_median > plain_median:
        return "loss", difference
    return "tie", difference


def paired_comparison(
    rows: list[dict[str, Any]], algorithm: str, seconds: float, delta_arm: str, plain_arm: str
) -> dict[str, float | str]:
    """Compare a delta arm with its plain arm instance by instance.

    Args:
        rows (list[dict[str, Any]]): Parsed JSONL rows.
        algorithm (str): Algorithm label used in both arms, for example `ILS`.
        seconds (float): Budget of the comparison.
        delta_arm (str): Arm key with the delta.
        plain_arm (str): Arm key without the delta.

    Returns:
        dict[str, float | str]: Algorithm, budget, wins, losses, ties, sign-test p-value and
        median relative difference
        of the median objective, where negative means the delta arm is better.
    """
    objectives = _objectives(rows, seconds, (delta_arm, plain_arm))
    outcomes = [
        _pair_outcome(plain, delta) for plain, delta in _matched(objectives, plain_arm, delta_arm)
    ]
    wins, losses, ties = _tally([outcome for outcome, _ in outcomes])
    relative = [difference for _, difference in outcomes]
    return {
        "algorithm": algorithm,
        "seconds": seconds,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "p_value": sign_test_p(wins, losses),
        "median_relative_difference": statistics.median(relative) if relative else 0.0,
    }


def render(rows: list[dict[str, Any]]) -> str:
    """Render arm summaries and paired comparisons as Markdown.

    Args:
        rows (list[dict[str, Any]]): Parsed JSONL rows.

    Returns:
        str: Markdown text with one arm table and one paired table.
    """
    lines = [
        "| Arm | Seconds | Median gap | Best gap | Median evals/s | Delta evals (share) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (arm, seconds), value in summarize_arms(rows).items():
        share = value["delta_evaluations"] / value["evaluations"] if value["evaluations"] else 0.0
        lines.append(
            f"| {arm} | {seconds:g} | {value['median_gap']:.4f} | {value['best_gap']:.4f} | "
            f"{value['median_evals_per_second']:.0f} | {share:.3f} |"
        )
    lines += [
        "",
        "| Algorithm | Seconds | Wins | Losses | Ties | Sign-test p | Median rel. diff |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for algorithm in ("default", "SA", "ILS"):
        for seconds in sorted({float(row["seconds"]) for row in rows}):
            result = paired_comparison(
                rows,
                algorithm,
                seconds,
                f"rust-{algorithm}+delta",
                f"rust-{algorithm}",
            )
            lines.append(
                f"| {algorithm} | {seconds:g} | {result['wins']} | {result['losses']} | "
                f"{result['ties']} | {result['p_value']:.3f} | "
                f"{result['median_relative_difference']:+.5f} |"
            )
    return "\n".join(lines)


def main() -> None:
    """Read a JSONL file and print the delta comparison tables.

    Raises:
        ValueError: If the file mixes provenance from different hosts, commits or code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path)
    arguments = parser.parse_args()
    rows = [json.loads(line) for line in arguments.rows.read_text(encoding="utf-8").splitlines()]
    provenance = {(row["host"], row["commit"], row["benchmark_sha256"]) for row in rows}
    if len(provenance) > 1:
        raise ValueError("Mixed provenance: compare one host, commit and benchmark revision.")
    print(render(rows))


if __name__ == "__main__":
    main()
