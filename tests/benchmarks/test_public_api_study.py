"""Tests for the whole-invocation overhead summary of the public API study.

Objective:
- Verify that the overhead of a complete `RKO.solve` call over its search time is computed and
  summarized from measured rows without changing the rows.

Technical approach:
- Feed small hand-made rows to the pure summary functions and check exact medians and fractions.

Flow (pseudocode):
1. Build rows with known elapsed and search time.
2. Compute the overhead fraction per row.
3. Check the per-configuration median summary, sequential only.

Important criteria:
- Overhead is solver elapsed time minus the summed search time, reported only for sequential runs.

Inputs/Outputs:
- Input: measured rows as dictionaries.
- Output: exact overhead values and summaries.
"""

import pytest
from benchmarks.runtime.public_api_study import overhead_fraction
from benchmarks.runtime.public_api_study import summarize


def test_overhead_fraction_is_the_share_outside_the_searches() -> None:
    """Overhead is the elapsed time not spent inside the searches themselves."""
    assert overhead_fraction(elapsed=2.0, search=1.5) == pytest.approx(0.25)


def test_overhead_fraction_rejects_nonpositive_elapsed_time() -> None:
    """A zero elapsed time has no defined fraction."""
    with pytest.raises(ValueError, match="elapsed"):
        overhead_fraction(elapsed=0.0, search=0.0)


def test_summary_reports_median_overhead_per_configuration() -> None:
    """Each configuration's median overhead comes from its own rows only."""
    rows = [
        {"config": "sequential", "elapsed": 1.5, "search": 1.0},
        {"config": "sequential", "elapsed": 1.0, "search": 0.5},
        {"config": "sequential", "elapsed": 3.0, "search": 2.0},
        {"config": "thread", "elapsed": 1.0, "search": 0.9},
    ]
    summary = summarize(rows)
    assert summary["sequential"]["n"] == 3
    assert summary["sequential"]["median_overhead_fraction"] == pytest.approx(1 / 3)
    assert "thread" not in summary


def test_overhead_fraction_is_defined_only_for_sequential_accounting() -> None:
    """Parallel search times overlap, so no sum-based overhead is reported for them."""
    rows = [{"config": "thread", "elapsed": 1.0, "search": 0.9}]
    assert summarize(rows) == {}
