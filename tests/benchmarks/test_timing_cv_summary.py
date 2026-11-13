"""Regression over the timed-run variance summary (issue 99).

Objective:
- Pin the per-instance coefficient of variation (CV) arithmetic and the committed
  reference medians that the benchmarks README reports.

Technical approach:
- Check the pure helpers on hand-built vectors, including the zero-mean NaN case and the
  two-repeat filter, then read the shipped reference files and assert the README medians.

Flow (pseudocode):
1. Check the CV helper on a known vector and on a zero mean.
2. Check that instances with fewer than two repeats never enter a group.
3. Read each committed reference file and assert its per-class medians and rate CV.

Important criteria:
- Only instances with at least two repeats enter a CV; a zero mean yields NaN.
- The asserted medians must match the README table exactly.

Inputs/Outputs:
- Input: hand-built rows and the committed `references/timing-cv-*.jsonl` files.
- Output: deterministic assertions.
"""

import json
import math
import pytest
import statistics
from benchmarks.algorithms.timing_cv_summary import coefficient_of_variation
from benchmarks.algorithms.timing_cv_summary import read_groups
from benchmarks.algorithms.timing_cv_summary import summarize
from benchmarks.references import store
from pathlib import Path


def test_cv_matches_population_definition() -> None:
    """Match the population standard deviation over the mean on a known vector.

    Returns:
        None: Assertion only.
    """
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    expected = statistics.pstdev(values) / statistics.fmean(values)
    assert coefficient_of_variation(values) == pytest.approx(expected)


def test_cv_is_nan_when_mean_is_zero() -> None:
    """Return NaN when the mean is zero so the group is skipped later.

    Returns:
        None: Assertion only.
    """
    assert math.isnan(coefficient_of_variation([-1.0, 1.0]))


def test_read_groups_keeps_timed_rows_and_time2_rates(tmp_path: Path) -> None:
    """Keep only timed profiles for evaluations and only time2 for rates.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertion only.
    """
    rows = [
        {
            "class": "item",
            "profile": "time2",
            "instance": "a",
            "evaluations": 100,
            "search_seconds": 2.0,
        },
        {
            "class": "item",
            "profile": "time2",
            "instance": "a",
            "evaluations": 200,
            "search_seconds": 2.0,
        },
        {
            "class": "item",
            "profile": "time10",
            "instance": "a",
            "evaluations": 300,
            "search_seconds": 10.0,
        },
        {
            "class": "item",
            "profile": "evaluations",
            "instance": "a",
            "evaluations": 400,
            "search_seconds": 1.0,
        },
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    evaluations, rates = read_groups(path)
    assert evaluations[("item", "time2", "a")] == [100.0, 200.0]
    assert evaluations[("item", "time10", "a")] == [300.0]
    assert ("item", "evaluations", "a") not in evaluations
    assert rates[("item", "a")] == [50.0, 100.0]


@pytest.mark.parametrize(
    ("name", "rate_cv", "medians"),
    [
        (
            "timing-cv-temperature-free-wsl2.jsonl",
            0.285,
            {
                ("continuous", "time2"): 0.179,
                ("continuous", "time10"): 0.135,
                ("item", "time2"): 0.432,
                ("item", "time10"): 0.146,
                ("route", "time2"): 0.285,
                ("route", "time10"): 0.230,
            },
        ),
        (
            "timing-cv-temperature-pinned-cpu1-wsl2.jsonl",
            0.170,
            {
                ("continuous", "time2"): 0.152,
                ("continuous", "time10"): 0.121,
                ("item", "time2"): 0.186,
                ("item", "time10"): 0.125,
                ("route", "time2"): 0.226,
                ("route", "time10"): 0.251,
            },
        ),
        (
            "timing-cv-adaptive-pinned-cpu1-wsl2.jsonl",
            0.245,
            {
                ("continuous", "time2"): 0.228,
                ("continuous", "time10"): 0.203,
                ("item", "time2"): 0.199,
                ("item", "time10"): 0.129,
                ("route", "time2"): 0.299,
                ("route", "time10"): 0.120,
            },
        ),
    ],
)
def test_reference_medians_match_readme(
    name: str, rate_cv: float, medians: dict[tuple[str, str], float]
) -> None:
    """Recompute the README medians from the committed reference data.

    Args:
        name (str): Reference file name under benchmarks/references.
        rate_cv (float): Expected median CV of evals/s for time2.
        medians (dict[tuple[str, str], float]): Expected per-class per-profile medians.

    Returns:
        None: Assertion only.
    """
    evaluations, rates = read_groups(store.ensure(name))
    groups: dict[tuple[str, str], list[float]] = {}
    for (cls, profile, _), values in evaluations.items():
        if len(values) >= 2:
            groups.setdefault((cls, profile), []).append(coefficient_of_variation(values))
    for key, expected in medians.items():
        assert statistics.median(groups[key]) == pytest.approx(expected, abs=5e-4)
    rate_cvs = [coefficient_of_variation(v) for v in rates.values() if len(v) >= 2]
    assert statistics.median(rate_cvs) == pytest.approx(rate_cv, abs=5e-4)


def test_summarize_prints_every_class_and_profile(capsys: pytest.CaptureFixture[str]) -> None:
    """Print one line per class and profile plus the rate CV for a reference file.

    Args:
        capsys (pytest.CaptureFixture[str]): Captured standard output.

    Returns:
        None: Assertion only.
    """
    summarize(store.ensure("timing-cv-temperature-free-wsl2.jsonl"))
    out = capsys.readouterr().out
    assert "timing-cv-temperature-free-wsl2.jsonl" in out
    assert out.count("median=") == 6
    assert "evals/s (time2) median CV=" in out
