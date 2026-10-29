"""Priority decoder study contract tests.

Objective:
- Check the instance generator, the exact reference and the paired statistics.

Technical approach:
- Compare the DP oracle with brute force and check reproducibility and bounds by construction.

Flow (pseudocode):
1. Generate instances and solve them exactly two ways.
2. Compare aligned gap vectors with known differences.
3. Check candidate construction errors.

Important criteria:
- Instances are reproducible and disjoint across seeds.
- Win, loss and tie counts add up to the number of instances.

Inputs/Outputs:
- Input: generated instances and gap vectors.
- Output: assertions on optima, statistics and errors.
"""

import itertools
import numpy as np
import pytest
from benchmarks.engines.priority_decoder_study import FAMILIES
from benchmarks.engines.priority_decoder_study import Candidate
from benchmarks.engines.priority_decoder_study import GreedySeed
from benchmarks.engines.priority_decoder_study import InstanceFactory
from benchmarks.engines.priority_decoder_study import PairedStatistics
from benchmarks.problems.knapsack_utils import exact_profit
from examples.priority_decoders import PriorityDecoder
from examples.priority_decoders import SortDecoder


def test_generator_is_reproducible_and_seed_sensitive() -> None:
    """Same arguments give the same instance and another seed gives another one."""
    for family in FAMILIES:
        first = InstanceFactory.make(family, 30, 5)
        assert first == InstanceFactory.make(family, 30, 5)
        assert first != InstanceFactory.make(family, 30, 6)
        assert min(first.benefits) >= 1 and min(first.weights) >= 1
        assert first.capacity >= max(first.weights)


def test_unknown_family_is_rejected() -> None:
    """Only the documented families are accepted."""
    with pytest.raises(ValueError, match="family"):
        InstanceFactory.make("random", 10, 1)


@pytest.mark.parametrize("family", FAMILIES)
def test_exact_reference_matches_brute_force(family: str) -> None:
    """The DP oracle equals exhaustive search on a 12-item instance.

    Args:
        family (str): Instance family.
    """
    instance = InstanceFactory.make(family, 12, 8)
    best = max(
        sum(b for b, pick in zip(instance.benefits, mask, strict=True) if pick)
        for mask in itertools.product((0, 1), repeat=12)
        if sum(w for w, pick in zip(instance.weights, mask, strict=True) if pick)
        <= instance.capacity
    )
    assert exact_profit(instance) == best


def test_paired_statistics_counts_and_signs() -> None:
    """Negative differences favor the candidate; counts partition the instances."""
    stats = PairedStatistics.compare([1.0, 2.0, 3.0, 3.0], [2.0, 2.0, 2.0, 4.0], 500, 0)
    assert (stats["wins"], stats["losses"], stats["ties"]) == (2, 1, 1)
    assert stats["difference"] == pytest.approx(-0.25)
    assert stats["ci_low"] <= stats["difference"] <= stats["ci_high"]


def test_candidate_requires_a_known_name_and_parameter() -> None:
    """Informed decoders need a parameter and unknown names are rejected."""
    instance = InstanceFactory.make("uncorrelated", 10, 1)
    with pytest.raises(ValueError, match="Unknown"):
        Candidate("priority").decoder(instance)
    with pytest.raises(ValueError, match="Unknown"):
        Candidate("other").decoder(instance)


@pytest.mark.parametrize("family", FAMILIES)
def test_greedy_seed_makes_the_sort_decoder_return_the_priority_solution(family: str) -> None:
    """The seed keys carry the same prior as the priority decoder at neutral keys.

    Args:
        family (str): Instance family.
    """
    instance = InstanceFactory.make(family, 60, 3)
    keys = GreedySeed.keys(instance)
    assert all(0.0 < key < 1.0 for key in keys)
    neutral = PriorityDecoder.build(instance, 1.0)(np.full(60, 0.5))
    assert SortDecoder(instance)(np.asarray(keys)) == neutral


def test_against_sort_groups_by_budget_and_includes_the_reference() -> None:
    """Each decoder is compared with sort at the selected budget only."""
    rows = [
        {"decoder": name, "budget": budget, "gap": gap}
        for budget in (1.0, 2.0)
        for name, gap in (("sort", 2.0), ("sort_seeded", 0.5))
        for _ in range(3)
    ]
    table = PairedStatistics.against_sort(rows, 1.0, 100)
    assert set(table) == {"sort@1s", "sort_seeded@1s"}
    assert table["sort@1s"]["difference"] == 0.0
    assert table["sort_seeded@1s"]["difference"] == pytest.approx(-1.5)
    assert table["sort_seeded@1s"]["wins"] == 3
