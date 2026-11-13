"""Paired, interleaved timing comparisons that stay valid on a shared host.

Objective:
- Compare two arms by time when background load on the host cannot be controlled, without
  requiring an idle machine.

Technical approach:
- Run the arms alternately (A, B, A, B, ...) so slow drift in load affects both arms alike.
- Require equal work per pair (the same evaluation count) before a time ratio means anything.
- Summarize the paired ratios by their geometric mean, a bootstrap interval over pairs and an
  exact two-sided sign test on the pair-wise winners.
- Read the CPU busy fraction from the kernel counters over each run, which is finer than the
  one-minute load average.

Flow (pseudocode):
1. Produce the alternating order of arms for the requested number of pairs.
2. Run each arm with a shared seed, recording time, work and load for every run.
3. Compute each pair's ratio and check equal work.
4. Summarize the ratios with an interval and a sign-test p-value.

Important criteria:
- The interval is over pairs, so a few noisy runs cannot create a claim by themselves.
- Pairs with different work are reported and excluded from the time claim.

Inputs/Outputs:
- Input: a runner that performs one run for an arm and a seed, and the number of pairs.
- Output: per-pair records, and a summary with the ratio, interval, p-value and load statistics.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable
from typing import Any

KERNEL_IDLE_FIELDS = (3, 4)


def interleaved_order(pairs: int) -> list[tuple[str, int]]:
    """Return the arm order for a number of pairs, alternating A and B within each pair.

    Args:
        pairs (int): Number of pairs; must be positive.

    Returns:
        list[tuple[str, int]]: (arm, pair index) in execution order.
    """
    return [(arm, index) for index in range(pairs) for arm in ("A", "B")]


def busy_fraction(before: str, after: str) -> float:
    """Return the share of CPU time that was busy between two `/proc/stat` readings.

    Args:
        before (str): Text of `/proc/stat` taken at the start of the run.
        after (str): Text of `/proc/stat` taken at the end of the run.

    Returns:
        float: Busy jiffies over all jiffies in the aggregate `cpu` line; NaN when unchanged.
    """
    first = _aggregate_cpu(before)
    last = _aggregate_cpu(after)
    delta = [later - earlier for earlier, later in zip(first, last, strict=True)]
    total = sum(delta)
    if total <= 0:
        return math.nan
    idle = sum(delta[index] for index in KERNEL_IDLE_FIELDS)
    return (total - idle) / total


def _aggregate_cpu(text: str) -> list[int]:
    """Parse the aggregate `cpu` line of `/proc/stat` into integer counters.

    Args:
        text (str): Full text of `/proc/stat`.

    Returns:
        list[int]: The counters of the aggregate line, in kernel order.

    Raises:
        ValueError: If the aggregate line is missing.
    """
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == "cpu":
            return [int(value) for value in fields[1:]]
    raise ValueError("no aggregate cpu line in /proc/stat text")


def sign_test_p(wins: int, losses: int) -> float:
    """Return the exact two-sided sign-test p-value for paired wins and losses.

    Args:
        wins (int): Pairs where the first arm was faster.
        losses (int): Pairs where the second arm was faster.

    Returns:
        float: Two-sided exact binomial p-value, capped at one.
    """
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, count) for count in range(min(wins, losses) + 1)) / 2**trials
    return min(1.0, 2.0 * tail)


def bootstrap_ratio_ci(
    ratios: list[float], rng: random.Random, samples: int = 5000
) -> tuple[float, float, float]:
    """Return the geometric-mean ratio and its 95% bootstrap interval over pairs.

    Args:
        ratios (list[float]): Positive paired time ratios, one per pair.
        rng (random.Random): Generator for the resampling.
        samples (int): Number of bootstrap resamples.

    Returns:
        tuple[float, float, float]: Geometric mean, lower and upper 95% bounds.

    Raises:
        ValueError: If there are no ratios or a ratio is not positive.
    """
    if not ratios or any(value <= 0 for value in ratios):
        raise ValueError("ratios must be a non-empty list of positive values")
    logs = [math.log(value) for value in ratios]
    means = []
    for _ in range(samples):
        draw = [logs[rng.randrange(len(logs))] for _ in logs]
        means.append(statistics.fmean(draw))
    means.sort()
    centre = math.exp(statistics.fmean(logs))
    return centre, math.exp(means[int(0.025 * samples)]), math.exp(means[int(0.975 * samples)])


def _pair_outcomes(rows: list[dict[str, dict[str, Any]]]) -> tuple[int, int]:
    """Count the pairs won by each arm.

    Args:
        rows (list[dict[str, dict[str, Any]]]): Paired run records.

    Returns:
        tuple[int, int]: Pairs where arm A was faster, and pairs where arm B was faster.
    """
    wins = sum(1 for row in rows if row["a"]["elapsed"] < row["b"]["elapsed"])
    losses = sum(1 for row in rows if row["b"]["elapsed"] < row["a"]["elapsed"])
    return wins, losses


def _load_statistics(rows: list[dict[str, dict[str, Any]]]) -> tuple[float, float]:
    """Return the median and maximum of every load reading taken around the runs.

    Args:
        rows (list[dict[str, dict[str, Any]]]): Paired run records.

    Returns:
        tuple[float, float]: Median and maximum one-minute load over all readings.
    """
    readings = [
        value
        for row in rows
        for run in (row["a"], row["b"])
        for value in (run["load_before"], run["load_after"])
    ]
    return float(statistics.median(readings)), float(max(readings))


def summarize_pairs(
    rows: list[dict[str, dict[str, Any]]],
    rng: random.Random,
) -> dict[str, Any]:
    """Summarize interleaved pairs: ratio, interval, sign test, equal-work check and load.

    Args:
        rows (list[dict[str, dict[str, Any]]]): One {"a": run, "b": run} record per pair, where a
            run has "elapsed", "evaluations", "load_before" and "load_after".
        rng (random.Random): Generator for the bootstrap.

    Returns:
        dict[str, Any]: Summary with the ratio, interval, coverage of one, sign-test p-value,
            equal-work flag and load statistics.
    """
    ratios = [row["b"]["elapsed"] / row["a"]["elapsed"] for row in rows]
    centre, low, high = bootstrap_ratio_ci(ratios, rng)
    wins, losses = _pair_outcomes(rows)
    load_median, load_max = _load_statistics(rows)
    return {
        "pairs": len(rows),
        "ratio_geometric_mean": centre,
        "ci95": [low, high],
        "covers_1": low <= 1.0 <= high,
        "sign_p": sign_test_p(wins, losses),
        "equal_work": all(row["a"]["evaluations"] == row["b"]["evaluations"] for row in rows),
        "load1_median": load_median,
        "load1_max": load_max,
    }


def run_interleaved(
    runner: Callable[[str, int], dict[str, Any]],
    pairs: int,
    seed_start: int = 0,
) -> list[dict[str, dict[str, Any]]]:
    """Run interleaved pairs with one seed per pair, shared by both arms.

    Args:
        runner (Callable[[str, int], dict[str, Any]]): Performs one run for an arm and a seed.
        pairs (int): Number of pairs; must be positive.
        seed_start (int): Seed of the first pair.

    Returns:
        list[dict[str, dict[str, Any]]]: One {"a": run, "b": run} record per pair.
    """
    records: list[dict[str, dict[str, Any]]] = [{} for _ in range(pairs)]
    for arm, index in interleaved_order(pairs):
        records[index][arm.lower()] = runner(arm, seed_start + index)
    return records
