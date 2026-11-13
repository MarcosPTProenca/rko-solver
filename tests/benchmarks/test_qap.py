"""Contract tests for the quadratic assignment (QAP) parser, decoder and objective.

Objective:
- Verify that QAPLIB-format text is parsed exactly, that the decoder yields a permutation, and that
  the objective equals an independent assignment cost on every small instance.

Technical approach:
- Use hand-written tiny instances and exhaustive permutations as the oracle, so no network is used.

Flow (pseudocode):
1. Parse a tiny instance with and without the enriched `SIZE OPT BKS` header.
2. Compare the objective with a direct sum over all facility pairs, for every permutation.
3. Check the decoder returns a permutation and rejects malformed input.

Important criteria:
- The objective is the sum of flow times distance over facility pairs for the decoded assignment.
- Malformed matrices and non-square input are rejected before any search starts.

Inputs/Outputs:
- Input: QAPLIB-format text and key vectors.
- Output: parsed instances, costs and permutations.
"""

import itertools
import numpy as np
import pytest
from benchmarks.public.qap import QapObjective
from benchmarks.public.qap import parse_qap

TINY_PLAIN = """3
0 5 2
5 0 3
2 3 0
0 1 2
1 0 3
2 3 0
"""

TINY_ENRICHED = "3 4 4\n" + TINY_PLAIN.split("\n", 1)[1]


def direct_cost(flow: np.ndarray, distance: np.ndarray, order: tuple[int, ...]) -> float:
    """Sum flow times distance over facility pairs for an assignment given as a facility order.

    Args:
        flow (np.ndarray): Facility flow matrix.
        distance (np.ndarray): Location distance matrix.
        order (tuple[int, ...]): Facility placed at each location, in location order.

    Returns:
        float: Assignment cost.
    """
    total = 0.0
    for k in range(len(order)):
        for m in range(len(order)):
            total += flow[order[k], order[m]] * distance[k, m]
    return total


def test_parses_plain_qaplib_text() -> None:
    """A plain instance yields its size and both matrices exactly."""
    instance = parse_qap(TINY_PLAIN)
    assert instance.size == 3
    assert instance.optimum is None
    assert instance.best_known is None
    assert instance.flow.tolist() == [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    assert instance.distance.tolist() == [[0, 1, 2], [1, 0, 3], [2, 3, 0]]


def test_parses_enriched_header_with_optimum_and_best_known() -> None:
    """The `SIZE OPT BKS` header records the optimum and the best-known value."""
    instance = parse_qap(TINY_ENRICHED)
    assert instance.optimum == 4
    assert instance.best_known == 4


def test_objective_matches_direct_cost_for_every_assignment() -> None:
    """For every decoded assignment the objective equals the direct pairwise sum."""
    instance = parse_qap(TINY_PLAIN)
    objective = QapObjective(instance.flow, instance.distance)
    for order in itertools.permutations(range(3)):
        keys = np.argsort(np.argsort(np.asarray(order, dtype=float))).astype(float)
        decoded = objective.decode(keys)
        assert sorted(decoded.tolist()) == [0, 1, 2]
        assert objective.cost(decoded) == pytest.approx(
            direct_cost(instance.flow, instance.distance, tuple(decoded.tolist()))
        )


def test_decoder_returns_a_permutation_for_any_keys() -> None:
    """Keys are sorted into a permutation, including ties, which are broken stably."""
    objective = QapObjective(np.zeros((4, 4)), np.zeros((4, 4)))
    decoded = objective.decode(np.array([0.5, 0.5, 0.1, 0.9]))
    assert sorted(decoded.tolist()) == [0, 1, 2, 3]
    assert decoded.tolist() == [2, 0, 1, 3]


@pytest.mark.parametrize(
    "text",
    [
        "3\n0 1\n1 0 2\n",
        "2\n0 1\n1 0\n0 1\n",
        "3\n0 1 2\n1 0 3\n2 3 0\n0 1 2\n1 0 3\n2 3 x\n",
        "2 1 2 3\n0 1\n1 0\n0 1\n1 0\n",
        "",
    ],
)
def test_malformed_input_is_rejected(text: str) -> None:
    """Missing, non-numeric or non-square input raises before any objective is built.

    Args:
        text (str): Malformed instance text.
    """
    with pytest.raises(ValueError):
        parse_qap(text)
