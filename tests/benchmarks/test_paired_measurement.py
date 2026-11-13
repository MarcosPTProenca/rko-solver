"""Contract tests for the paired, interleaved measurement harness.

Objective:
- Verify the design choices that make comparisons valid on a shared host: alternating arms,
  equal-work checks, CPU busy fraction from the kernel counters, an exact sign test and a
  bootstrap interval over pairs.

Technical approach:
- Use synthetic timings and fixture text for `/proc/stat`, so every check is deterministic and
  needs no real load.

Flow (pseudocode):
1. Check arm interleaving and the busy-fraction parser on fixed text.
2. Check the exact sign test and the bootstrap interval on constructed ratios.
3. Check the null and known-effect validation on seeded synthetic runs.

Important criteria:
- A null comparison must cover 1; a constant injected effect must be detected.
- Pairs with different evaluation counts must not pass the equal-work check.

Inputs/Outputs:
- Input: arm labels, counter text and per-pair run records.
- Output: orderings, fractions, p-values, intervals and summaries.
"""

import pytest
import random
from benchmarks.measurement.paired import bootstrap_ratio_ci
from benchmarks.measurement.paired import busy_fraction
from benchmarks.measurement.paired import interleaved_order
from benchmarks.measurement.paired import sign_test_p
from benchmarks.measurement.paired import summarize_pairs
from collections.abc import Callable

STAT_BEFORE = "cpu  100 0 100 800 0 0 0 0 0 0\ncpu0 1 0 1 1 0 0 0 0 0 0\n"
STAT_AFTER = "cpu  200 0 150 900 0 0 0 0 0 0\ncpu0 1 0 1 1 0 0 0 0 0 0\n"


def test_arms_alternate_within_each_pair() -> None:
    """Each pair runs A then B, so drift in host load affects both arms alike."""
    assert interleaved_order(2) == [("A", 0), ("B", 0), ("A", 1), ("B", 1)]


def test_busy_fraction_counts_non_idle_jiffies_between_two_samples() -> None:
    """Busy time is the change in all counters except idle and iowait, over the total change."""
    busy = 100 + 50
    total = busy + 100
    assert busy_fraction(STAT_BEFORE, STAT_AFTER) == pytest.approx(busy / total)


def test_sign_test_is_exact_and_two_sided() -> None:
    """Ten wins and no losses give the exact two-sided binomial probability."""
    assert sign_test_p(10, 0) == pytest.approx(2 * 0.5**10)
    assert sign_test_p(0, 0) == 1.0


def test_bootstrap_interval_covers_one_for_symmetric_noise() -> None:
    """Ratios scattered around one give an interval that includes one."""
    rng = random.Random(3)
    ratios = [1.0 + rng.uniform(-0.03, 0.03) for _ in range(30)]
    _, low, high = bootstrap_ratio_ci(ratios, random.Random(1))
    assert low <= 1.0 <= high


def test_bootstrap_interval_excludes_one_for_a_constant_effect() -> None:
    """A consistent 20% slowdown with little noise gives an interval above one."""
    rng = random.Random(4)
    ratios = [1.2 + rng.uniform(-0.01, 0.01) for _ in range(30)]
    _, low, _ = bootstrap_ratio_ci(ratios, random.Random(1))
    assert low > 1.0


def _synthetic_runner(effect: float) -> Callable[[str, int], dict[str, object]]:
    """Build a runner whose time depends on the arm and a seed-derived noise term.

    Args:
        effect (float): Multiplicative slowdown applied to arm B.

    Returns:
        Callable: A function taking (arm, seed) and returning a run record.
    """

    def run(arm: str, seed: int) -> dict[str, object]:
        """Return one synthetic run record for an arm and seed.

        Args:
            arm (str): "A" or "B".
            seed (int): Seed that fixes the noise of the run.

        Returns:
            dict[str, object]: Elapsed time, evaluations and load readings.
        """
        noise = random.Random(seed * 7 + (1 if arm == "B" else 0)).uniform(-0.02, 0.02)
        base = 1.0 + noise
        elapsed = base * (effect if arm == "B" else 1.0)
        return {
            "elapsed": elapsed,
            "evaluations": 20000,
            "load_before": 3.8,
            "load_after": 3.9,
        }

    return run


def test_null_comparison_has_no_false_difference() -> None:
    """Identical arms give a summary whose interval covers one."""
    rows = [{"a": r["a"], "b": r["b"]} for r in _pairs(_synthetic_runner(1.0), 20)]
    summary = summarize_pairs(rows, random.Random(2))
    assert summary["covers_1"] is True
    assert summary["equal_work"] is True


def test_known_effect_is_detected() -> None:
    """A 37% injected slowdown is found with an interval that excludes one."""
    rows = _pairs(_synthetic_runner(1.37), 20)
    summary = summarize_pairs(rows, random.Random(2))
    assert summary["covers_1"] is False
    assert summary["ratio_geometric_mean"] == pytest.approx(1.37, rel=0.03)


def test_different_evaluation_counts_fail_the_equal_work_check() -> None:
    """Pairs that did not perform the same work cannot be compared by time."""
    rows = _pairs(_synthetic_runner(1.0), 3)
    rows[1]["b"]["evaluations"] = 19999
    assert summarize_pairs(rows, random.Random(2))["equal_work"] is False


def _pairs(
    runner: Callable[[str, int], dict[str, object]], count: int
) -> list[dict[str, dict[str, object]]]:
    """Run interleaved synthetic pairs and collect them.

    Args:
        runner: Function taking (arm, seed) and returning a run record.
        count (int): Number of pairs.

    Returns:
        list[dict[str, dict[str, object]]]: One {"a": ..., "b": ...} record per pair.
    """
    rows = []
    for index in range(count):
        a = runner("A", index)
        b = runner("B", index)
        rows.append({"a": a, "b": b})
    return rows
