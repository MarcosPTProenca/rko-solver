"""Priority and quantile Knapsack decoder contract tests.

Objective:
- Check that the analytic decoders are feasible, deterministic and consistent with their
  documented limits.

Technical approach:
- Compare against independent greedy computations and a plain argsort on random keys.
- Check the inverse-CDF draw on hand-computed intervals.

Flow (pseudocode):
1. Build small instances from the study generator.
2. Decode neutral, random and extreme keys.
3. Compare each result with the independent reference.

Important criteria:
- Every decoded selection fits the capacity.
- Keys at 0.5 reproduce the greedy value/weight solution.
- A very large spread reproduces the sort decoder.

Inputs/Outputs:
- Input: generated instances and key vectors.
- Output: assertions on selections, orders and validation errors.
"""

import numpy as np
import pickle
import pytest
from benchmarks.engines.priority_decoder_study import FAMILIES
from benchmarks.engines.priority_decoder_study import InstanceFactory
from examples.priority_decoders import GreedyFill
from examples.priority_decoders import PriorityDecoder
from examples.priority_decoders import QuantileCodec
from examples.priority_decoders import QuantileDecoder
from examples.priority_decoders import SortDecoder
from examples.run_kp_cpp_instances import KpInstance
from examples.run_kp_cpp_instances import evaluate_solution


def greedy_by_ratio(instance: KpInstance) -> list[int]:
    """Compute the value/weight greedy selection without the decoder machinery.

    Args:
        instance (KpInstance): Knapsack data.

    Returns:
        list[int]: Binary selection filling items by descending ratio, later index first on ties.
    """
    order = sorted(
        range(instance.n_items),
        key=lambda index: (-instance.benefits[index] / instance.weights[index], -index),
    )
    return GreedyFill.select(np.asarray(order), instance)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_decoder_returns_feasible_selections(family: str) -> None:
    """Decode random keys and check capacity feasibility.

    Args:
        family (str): Instance family.
    """
    instance = InstanceFactory.make(family, 40, 7)
    decoders = [
        SortDecoder(instance),
        PriorityDecoder.build(instance, 0.5),
        QuantileDecoder.build(instance, 0.1),
    ]
    rng = np.random.default_rng(1)
    for decoder in decoders:
        for _ in range(20):
            _, _, feasible = evaluate_solution(decoder(rng.random(40)), instance)
            assert feasible


@pytest.mark.parametrize("family", FAMILIES)
def test_neutral_keys_reproduce_the_greedy_ratio_solution(family: str) -> None:
    """Keys at 0.5 must select the same items as the independent greedy computation.

    Args:
        family (str): Instance family.
    """
    instance = InstanceFactory.make(family, 60, 3)
    decoded = PriorityDecoder.build(instance, 1.0)(np.full(60, 0.5))
    assert decoded == greedy_by_ratio(instance)


def test_large_spread_matches_the_sort_decoder() -> None:
    """A dominant residual makes the priority decoder order items by key alone."""
    instance = InstanceFactory.make("uncorrelated", 50, 11)
    rng = np.random.default_rng(5)
    for _ in range(10):
        keys = rng.random(50)
        assert PriorityDecoder.build(instance, 1e9)(keys) == SortDecoder(instance)(keys)


def test_quantile_draw_follows_the_inverse_cdf_and_skips_zero_weights() -> None:
    """Draw from weights [1, 0, 3]: item 0 owns [0, 0.25), item 2 owns [0.25, 1)."""
    weights = np.array([1.0, 0.0, 3.0])
    assert QuantileCodec.draw(0.0, weights) == 0
    assert QuantileCodec.draw(0.24, weights) == 0
    assert QuantileCodec.draw(0.26, weights) == 2
    assert QuantileCodec.draw(0.999, weights) == 2


def test_quantile_order_is_a_permutation() -> None:
    """Every item is drawn exactly once, including for extreme keys."""
    instance = InstanceFactory.make("weakly_correlated", 30, 2)
    scores = np.linspace(0.0, 1.0, 30)
    for keys in (np.zeros(30), np.full(30, 0.999999), np.random.default_rng(0).random(30)):
        order = QuantileCodec.order(keys, scores, 0.1)
        assert sorted(order.tolist()) == list(range(instance.n_items))


def test_decoders_are_pickleable_and_deterministic() -> None:
    """Spawned workers receive the decoder by pickle and must decode identically."""
    instance = InstanceFactory.make("strongly_correlated", 25, 4)
    keys = np.random.default_rng(9).random(25)
    for decoder in (
        SortDecoder(instance),
        PriorityDecoder.build(instance, 0.5),
        QuantileDecoder.build(instance, 0.3),
    ):
        assert pickle.loads(pickle.dumps(decoder))(keys) == decoder(keys)


def test_nonpositive_hyperparameters_are_rejected() -> None:
    """Zero spread or temperature is invalid."""
    instance = InstanceFactory.make("uncorrelated", 10, 1)
    with pytest.raises(ValueError, match="spread"):
        PriorityDecoder.build(instance, 0.0)
    with pytest.raises(ValueError, match="temperature"):
        QuantileDecoder.build(instance, 0.0)
