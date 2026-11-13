"""Checks for the public delta comparison report (issue 157).

Objective:
- Keep the paired counts, sign-test p-values and provenance guard of the delta report correct.

Technical approach:
- Use hand-computed sign-test values and a tiny inline set of paired rows.

Flow (pseudocode):
1. Compare exact sign-test values with hand-computed probabilities.
2. Pair one instance where the delta arm is better and one where it is worse.
3. Check the median relative difference and the rendered header.

Important criteria:
- No file or network access; the expected numbers are written out by hand.

Inputs/Outputs:
- Input: inline rows.
- Output: assertions.
"""

from benchmarks.public.delta_report import paired_comparison
from benchmarks.public.delta_report import render
from benchmarks.public.delta_report import sign_test_p


def row(instance: str, arm: str, objective: float, seed: int) -> dict[str, object]:
    """Build one minimal row with the fields the report reads."""
    return {
        "instance": instance,
        "arm": arm,
        "seconds": 10.0,
        "seed": seed,
        "objective": objective,
        "gap": 0.0,
        "evaluations": 100,
        "wall_seconds": 1.0,
        "delta_evaluations": 0,
        "host": "h",
        "commit": "c",
        "benchmark_sha256": "d",
    }


def test_sign_test_matches_hand_computed_values() -> None:
    """Five of five on one side gives 2/32; three against seven gives 0.34375."""
    assert sign_test_p(0, 5) == 0.0625
    assert abs(sign_test_p(3, 7) - 0.34375) < 1e-12
    assert sign_test_p(0, 0) == 1.0


def test_paired_comparison_counts_wins_losses_and_ties() -> None:
    """Medians over seeds decide each pair; the delta arm wins one instance and loses one."""
    rows = [
        row("a", "rust-ILS", 100.0, 0),
        row("a", "rust-ILS", 102.0, 1),
        row("a", "rust-ILS+delta", 98.0, 0),
        row("a", "rust-ILS+delta", 99.0, 1),
        row("b", "rust-ILS", 50.0, 0),
        row("b", "rust-ILS+delta", 55.0, 0),
    ]
    result = paired_comparison(rows, "ILS", 10.0, "rust-ILS+delta", "rust-ILS")
    assert (result["wins"], result["losses"], result["ties"]) == (1, 1, 0)
    assert (
        result["median_relative_difference"] == ((98.5 - 101.0) / 101.0 + (55.0 - 50.0) / 50.0) / 2
    )


def test_render_lists_each_arm_and_every_algorithm() -> None:
    """The rendered text contains the arm table header and one paired line per algorithm."""
    rows = [row("a", "rust-ILS", 100.0, 0), row("a", "rust-ILS+delta", 99.0, 0)]
    text = render(rows)
    assert "| Arm | Seconds | Median gap |" in text
    assert text.count("| ILS | 10 |") == 1
    assert text.count("| SA | 10 |") == 1
