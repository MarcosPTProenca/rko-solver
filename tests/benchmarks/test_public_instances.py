"""Checks for the public-instance benchmark parsers, gap formula and report (issue 156).

Objective:
- Keep the parsers, the independent checker and the report arithmetic correct on tiny inline
  instances, with no network access.

Technical approach:
- Parse a three-city TSPLIB text and a two-instance OR-Library text written inline.
- Compare the checker's objective and gap with values computed by hand.
- Summarize a small fixture of rows and compare the reported medians.

Flow (pseudocode):
1. Parse inline text and assert sizes and optimum handling.
2. Build a problem and check known keys for objective, feasibility and gap.
3. Summarize rows and assert counts and medians.

Important criteria:
- No test touches the network or the cache.
- Expected values are written out by hand, not taken from the code under test.

Inputs/Outputs:
- Input: inline text fixtures.
- Output: assertions.
"""

import numpy as np
from benchmarks.public.parsers import PublicParsers
from benchmarks.public.problems import KnapsackObjective
from benchmarks.public.problems import PublicProblem
from benchmarks.public.report import render
from benchmarks.public.report import summarize

TSP_TEXT = """NAME : tri
TYPE : TSP
DIMENSION : 3
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 0 0
2 3 0
3 0 4
EOF
"""

MDKP_TEXT = """2
2 1 10
3 4
2 3
5
2 1 0
1 2
1 1
2
"""


def test_tsplib_rounds_distances_to_nearest_integer() -> None:
    """Distances use floor(x + 0.5): the 3-4-5 triangle gives 3, 4 and 5."""
    instance = PublicParsers.tsplib("tri", TSP_TEXT)
    assert instance.distances.shape == (3, 3)
    assert instance.distances[0, 1] == 3.0
    assert instance.distances[0, 2] == 4.0
    assert instance.distances[1, 2] == 5.0


def test_tsplib_rejects_other_edge_weight_types() -> None:
    """Only EUC_2D is accepted; anything else raises."""
    text = TSP_TEXT.replace("EUC_2D", "GEO")
    try:
        PublicParsers.tsplib("tri", text)
    except ValueError as error:
        assert "EUC_2D" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_solutions_take_first_numeric_token() -> None:
    """Annotations after the value, such as (CEIL_2D), are ignored."""
    assert PublicParsers.solutions("a280 : 2579\nlin : 18660188 (CEIL_2D)\n") == {
        "a280": 2579.0,
        "lin": 18660188.0,
    }


def test_mdkp_reads_profits_weights_capacities_and_zero_optimum_as_none() -> None:
    """A declared optimum of 0 means unknown; the rest is read in the documented order."""
    first, second = PublicParsers.mdkp("toy", MDKP_TEXT)
    assert first.name == "toy#1"
    assert first.profits.tolist() == [3.0, 4.0]
    assert first.weights.tolist() == [[2.0, 3.0]]
    assert first.capacities.tolist() == [5.0]
    assert first.optimum == 10.0
    assert second.profits.tolist() == [1.0, 2.0]
    assert second.optimum is None


def test_checker_gap_is_positive_when_worse_for_maximization() -> None:
    """Profit 3 against a reference of 4 is a gap of (4 - 3) / 4 = 0.25."""
    objective = KnapsackObjective(np.array([3.0, 4.0]), np.array([[2.0, 3.0]]), np.array([3.0]))
    problem = PublicProblem("toy", "orlib-mknap1", objective, 4.0)
    checked = problem.check(np.array([0.9, 0.1]))
    assert checked.objective == 3.0
    assert checked.feasible is True
    assert checked.gap == 0.25


def test_checker_reports_infeasible_selection() -> None:
    """Selecting both items exceeds capacity 3, so the selection is infeasible."""
    objective = KnapsackObjective(np.array([3.0, 4.0]), np.array([[2.0, 3.0]]), np.array([3.0]))
    problem = PublicProblem("toy", "orlib-mknap1", objective, 4.0)
    checked = problem.check(np.array([0.9, 0.9]))
    assert checked.feasible is False


def test_tsp_gap_is_positive_when_longer_than_reference() -> None:
    """A tour of length 10 against an optimum of 8 gives (10 - 8) / 8 = 0.25."""
    distances = np.array([[0.0, 2.0, 4.0], [2.0, 0.0, 4.0], [4.0, 4.0, 0.0]])
    from benchmarks.public.problems import TspObjective

    problem = PublicProblem("tri", "tsplib", TspObjective(distances), 8.0)
    checked = problem.check(np.array([0.1, 0.2, 0.3]))
    assert checked.objective == 10.0
    assert checked.gap == 0.25


def test_report_medians_and_counts() -> None:
    """Two feasible rows with gaps 0.1 and 0.3 give median 0.2; one infeasible row is counted."""
    rows = [
        {
            "family": "f",
            "arm": "a",
            "seconds": 10.0,
            "feasible": True,
            "gap": 0.1,
            "evaluations": 100,
            "wall_seconds": 10.0,
        },
        {
            "family": "f",
            "arm": "a",
            "seconds": 10.0,
            "feasible": True,
            "gap": 0.3,
            "evaluations": 300,
            "wall_seconds": 10.0,
        },
        {
            "family": "f",
            "arm": "a",
            "seconds": 10.0,
            "feasible": False,
            "gap": None,
            "evaluations": 200,
            "wall_seconds": 10.0,
        },
    ]
    stats = summarize(rows)[("f", "a", 10.0)]
    assert stats["rows"] == 3
    assert stats["feasible"] == 2
    assert stats["infeasible"] == 1
    assert abs(stats["median_gap"] - 0.2) < 1e-12
    assert abs(stats["best_gap"] - 0.1) < 1e-12
    assert "| a | 10 | 3 | 2 |" in render(summarize(rows))
