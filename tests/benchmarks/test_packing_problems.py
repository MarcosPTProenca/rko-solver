"""Packing instance, fill and decoder contract tests.

Objective:
- Verify feasibility, reproducibility and the order semantics the learned-priority study relies on.

Technical approach:
- Compare greedy fill and decoders with independent brute-force computations on small instances.

Flow (pseudocode):
1. Generate instances of both kinds.
2. Decode neutral, random and extreme keys.
3. Compare against independent references and check pickling.

Important criteria:
- Every selection satisfies all constraints.
- Neutral keys reproduce the score order; a huge spread reproduces the sort decoder.

Inputs/Outputs:
- Input: generated instances and key vectors.
- Output: assertions on selections, profits and errors.
"""

import numpy as np
import pickle
import pytest
from benchmarks.engines.packing_problems import PackingCost
from benchmarks.engines.packing_problems import PackingDecoder
from benchmarks.engines.packing_problems import PackingFamily
from benchmarks.engines.packing_problems import PackingInstance
from benchmarks.engines.packing_problems import PackingRun
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st


def feasible(selection: np.ndarray, instance: PackingInstance) -> bool:
    """Check every constraint independently of the code under test.

    Args:
        selection (np.ndarray): Binary selection.
        instance (PackingInstance): Packing instance.

    Returns:
        bool: True when all constraints hold.
    """
    return bool(np.all(instance.weights @ selection <= instance.capacities + 1e-9))


@pytest.mark.parametrize("kind", PackingFamily.KINDS)
def test_instances_are_reproducible_and_well_formed(kind: str) -> None:
    """Same arguments give the same data; another seed gives different data.

    Args:
        kind (str): Instance kind.
    """
    first = PackingFamily.make(kind, 40, 5)
    again = PackingFamily.make(kind, 40, 5)
    other = PackingFamily.make(kind, 40, 6)
    np.testing.assert_array_equal(first.profits, again.profits)
    np.testing.assert_array_equal(first.weights, again.weights)
    assert not np.array_equal(first.profits, other.profits)
    assert first.n_items == 40 and first.weights.shape == (len(first.capacities), 40)
    assert np.all(first.profits > 0) and np.all(first.weights >= 0) and np.all(first.capacities > 0)
    assert (len(first.capacities), kind) in ((1, "knapsack"), (5, "mkp"))


def test_unknown_kind_and_empty_size_are_rejected() -> None:
    """Only documented kinds and positive sizes are accepted."""
    with pytest.raises(ValueError, match="kind"):
        PackingFamily.make("bin", 10, 1)
    with pytest.raises(ValueError, match="n_items"):
        PackingFamily.make("mkp", 0, 1)


@pytest.mark.parametrize("kind", PackingFamily.KINDS)
def test_fill_matches_an_independent_greedy_and_is_feasible(kind: str) -> None:
    """Fill takes exactly the items a naive loop would take.

    Args:
        kind (str): Instance kind.
    """
    instance = PackingFamily.make(kind, 30, 2)
    order = np.random.default_rng(1).permutation(30)
    selection, profit = PackingFamily.fill(order, instance)
    used = np.zeros(len(instance.capacities))
    expected = np.zeros(30)
    for item in order:
        if np.all(used + instance.weights[:, item] <= instance.capacities):
            used += instance.weights[:, item]
            expected[item] = 1
    np.testing.assert_array_equal(selection, expected)
    assert feasible(selection, instance)
    assert profit == pytest.approx(float(instance.profits @ expected))


@settings(max_examples=40, derandomize=True, deadline=None)
@given(st.integers(min_value=0, max_value=10_000), st.sampled_from(PackingFamily.KINDS))
def test_decoders_always_return_feasible_selections(seed: int, kind: str) -> None:
    """Random keys and scores never produce an infeasible selection.

    Args:
        seed (int): Generation seed.
        kind (str): Instance kind.
    """
    instance = PackingFamily.make(kind, 25, seed)
    rng = np.random.default_rng(seed)
    for decoder in (
        PackingDecoder(instance),
        PackingDecoder(instance, rng.normal(size=25), 0.5),
    ):
        assert feasible(decoder(rng.random(25)), instance)


@pytest.mark.parametrize("kind", PackingFamily.KINDS)
def test_neutral_keys_reproduce_the_score_order(kind: str) -> None:
    """Keys at one half give the greedy fill in descending score order.

    Args:
        kind (str): Instance kind.
    """
    instance = PackingFamily.make(kind, 30, 3)
    scores = PackingFamily.hand_scores(instance)
    decoded = PackingDecoder(instance, scores, 1.0)(np.full(30, 0.5))
    expected, _ = PackingFamily.fill(np.argsort(-scores, kind="stable"), instance)
    np.testing.assert_array_equal(decoded, expected)


def test_huge_spread_matches_the_plain_sort_decoder() -> None:
    """A dominant residual makes the scored decoder order items by key alone."""
    instance = PackingFamily.make("mkp", 30, 4)
    scores = np.random.default_rng(2).normal(size=30)
    keys = np.random.default_rng(3).random(30)
    np.testing.assert_array_equal(
        PackingDecoder(instance, scores, 1e9)(keys), PackingDecoder(instance)(keys)
    )


@pytest.mark.parametrize("kind", PackingFamily.KINDS)
def test_order_keys_makes_the_sort_decoder_follow_the_scores(kind: str) -> None:
    """Encoded order keys reproduce the score-order fill through the plain sort decoder.

    Args:
        kind (str): Instance kind.
    """
    instance = PackingFamily.make(kind, 30, 7)
    scores = PackingFamily.hand_scores(instance)
    keys = np.asarray(PackingFamily.order_keys(scores))
    assert np.all(keys > 0.0) and np.all(keys < 1.0)
    expected, _ = PackingFamily.fill(np.argsort(-scores, kind="stable"), instance)
    np.testing.assert_array_equal(PackingDecoder(instance)(keys), expected)


@pytest.mark.parametrize("kind", PackingFamily.KINDS)
def test_features_and_hand_prior_have_the_documented_shapes(kind: str) -> None:
    """Features carry one row per item; the hand prior is finite and positive.

    Args:
        kind (str): Instance kind.
    """
    instance = PackingFamily.make(kind, 30, 8)
    features = PackingFamily.features(instance)
    assert features.shape == (30, 2 + len(instance.capacities))
    assert np.isfinite(features).all()
    prior = PackingFamily.hand_scores(instance)
    assert prior.shape == (30,) and np.all(np.isfinite(prior)) and np.all(prior > 0)


def test_cost_is_negative_profit_and_rejects_infeasible_selections() -> None:
    """The objective never accepts a selection that breaks a constraint."""
    instance = PackingFamily.make("mkp", 20, 1)
    selection, profit = PackingFamily.fill(np.arange(20), instance)
    assert PackingCost(instance)(selection) == pytest.approx(-profit)
    with pytest.raises(ValueError, match="constraint"):
        PackingCost(instance)(np.ones(20))
    with pytest.raises(ValueError, match="length"):
        PackingCost(instance)(np.ones(3))


def test_decoder_and_cost_survive_pickling() -> None:
    """Spawned workers receive the callbacks through pickle."""
    instance = PackingFamily.make("mkp", 20, 9)
    decoder = PackingDecoder(instance, PackingFamily.hand_scores(instance), 0.5)
    keys = np.random.default_rng(4).random(20)
    clone = pickle.loads(pickle.dumps((decoder, PackingCost(instance))))
    np.testing.assert_array_equal(clone[0](keys), decoder(keys))


def test_ten_constraint_family_is_well_formed_and_reproducible() -> None:
    """The harder family has ten constraints, tightness 0.5 and independent profits."""
    first = PackingFamily.make("mkp10", 60, 4)
    again = PackingFamily.make("mkp10", 60, 4)
    np.testing.assert_array_equal(first.profits, again.profits)
    np.testing.assert_array_equal(first.weights, again.weights)
    assert first.weights.shape == (10, 60) and np.all(first.profits > 0)
    np.testing.assert_allclose(first.capacities, 0.5 * first.weights.sum(axis=1))
    correlation = np.corrcoef(first.profits, first.weights.mean(axis=0))[0, 1]
    assert abs(correlation) < 0.5
    assert "mkp10" not in PackingFamily.KINDS


@settings(max_examples=30, derandomize=True, deadline=None)
@given(st.integers(min_value=0, max_value=5000), st.sampled_from(("mkp", "mkp10", "knapsack")))
def test_dual_bound_is_valid_and_multipliers_are_nonnegative(seed: int, kind: str) -> None:
    """Weak duality: the bound is at least the profit of any feasible selection.

    Args:
        seed (int): Generation seed.
        kind (str): Instance kind.
    """
    instance = PackingFamily.make(kind, 40, seed)
    multipliers, bound = PackingFamily.dual(instance)
    assert multipliers.shape == (len(instance.capacities),) and np.all(multipliers >= 0.0)
    rng = np.random.default_rng(seed)
    for _ in range(5):
        _, profit = PackingFamily.fill(rng.permutation(40), instance)
        assert bound >= profit - 1e-6


def test_dual_search_improves_on_its_starting_multipliers() -> None:
    """The returned bound is no worse than the bound at uniform unit multipliers."""
    instance = PackingFamily.make("mkp", 80, 3)
    scaled = instance.weights / instance.capacities[:, None]
    start = np.ones(len(instance.capacities))
    reduced = np.maximum(0.0, instance.profits - start @ scaled)
    assert PackingFamily.dual(instance)[1] <= float(start.sum() + reduced.sum()) + 1e-9


@pytest.mark.parametrize("kind", [*PackingFamily.KINDS, *PackingFamily.HARD_KINDS])
def test_dual_scores_are_positive_finite_and_deterministic(kind: str) -> None:
    """Dual scores never produce a zero, infinite or run-dependent priority.

    Args:
        kind (str): Instance kind.
    """
    instance = PackingFamily.make(kind, 50, 6)
    first = PackingFamily.dual_scores(instance)
    assert first.shape == (50,) and np.all(first > 0) and np.isfinite(first).all()
    np.testing.assert_array_equal(first, PackingFamily.dual_scores(instance))


def test_dual_prior_matches_or_beats_the_ratio_prior_on_the_correlated_family() -> None:
    """On average the stronger prior's greedy fill is at least as good as the ratio rule's."""
    gains = []
    for seed in range(20):
        instance = PackingFamily.make("mkp", 120, seed)
        ratio = PackingFamily.fill(np.argsort(-PackingFamily.hand_scores(instance)), instance)[1]
        dual = PackingFamily.fill(np.argsort(-PackingFamily.dual_scores(instance)), instance)[1]
        gains.append(dual / ratio)
    assert np.mean(gains) >= 1.0


def test_perturbed_keys_start_with_the_exact_order_and_are_reproducible() -> None:
    """The first vector encodes the exact order; noise makes the others distinct."""
    instance = PackingFamily.make("mkp", 40, 2)
    scores = PackingFamily.dual_scores(instance)
    keys = PackingFamily.perturbed_keys(scores, 8, 0.3, 5)
    assert keys[0] == PackingFamily.order_keys(scores)
    assert len(keys) == 8 and keys == PackingFamily.perturbed_keys(scores, 8, 0.3, 5)
    assert len({tuple(row) for row in keys}) == 8
    assert all(0.0 < key < 1.0 for row in keys for key in row)
    assert len({tuple(row) for row in PackingFamily.perturbed_keys(scores, 5, 0.0, 5)}) == 1


def test_perturbed_keys_reject_invalid_arguments() -> None:
    """A nonpositive count or negative noise is an error."""
    scores = np.arange(1.0, 6.0)
    with pytest.raises(ValueError, match="count"):
        PackingFamily.perturbed_keys(scores, 0, 0.1, 1)
    with pytest.raises(ValueError, match="noise"):
        PackingFamily.perturbed_keys(scores, 2, -0.1, 1)


def test_shared_solver_call_returns_a_verified_profit_and_honours_seed_keys() -> None:
    """One short real solve through the shared runner decodes and verifies its own result."""
    instance = PackingFamily.make("mkp", 24, 5)
    scores = PackingFamily.dual_scores(instance)
    keys = [PackingFamily.order_keys(scores)]
    decoder = PackingDecoder(instance)
    start = PackingFamily.fill(np.argsort(-scores, kind="stable"), instance)[1]
    profit, verified = PackingRun.solve(instance, decoder, keys, (3, 0.2), ("sequential", ("SA",)))
    assert verified and profit >= start - 1e-9
    assert PackingRun.solve(instance, decoder, None, (3, 0.2), ("sequential", ("SA",)))[1]
