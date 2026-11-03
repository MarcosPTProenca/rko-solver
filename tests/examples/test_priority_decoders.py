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
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st


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


@pytest.mark.parametrize("temperature", [1e-6, np.nextafter(0.0, 1.0)])
def test_extreme_temperature_never_repeats_a_selected_item(temperature: float) -> None:
    """Keep selecting remaining items even when their initial weights underflow.

    Args:
        temperature (float): Positive finite temperature near the numerical endpoint.
    """
    order = QuantileCodec.order(np.zeros(3), np.array([0.0, 0.5, 1.0]), temperature)
    assert order.tolist() == [2, 1, 0]


@pytest.mark.parametrize("key", [-0.1, 1.0, float("nan"), float("inf")])
def test_quantile_draw_rejects_invalid_keys(key: float) -> None:
    """Reject values outside the finite half-open key domain.

    Args:
        key (float): Invalid candidate key.
    """
    with pytest.raises(ValueError, match="key"):
        QuantileCodec.draw(key, np.ones(2))


@pytest.mark.parametrize("weights", [[], [0.0, 0.0], [-1.0, 2.0], [float("nan"), 1.0]])
def test_quantile_draw_rejects_invalid_weights(weights: list[float]) -> None:
    """Reject missing, negative, nonfinite or entirely zero weights.

    Args:
        weights (list[float]): Invalid categorical weights.
    """
    with pytest.raises(ValueError, match="weights"):
        QuantileCodec.draw(0.5, np.asarray(weights))


def test_quantile_draw_avoids_overflow_and_trailing_zero_categories() -> None:
    """Finite large weights still select a positive-weight category at the upper key endpoint."""
    assert QuantileCodec.draw(np.nextafter(1.0, 0.0), np.array([1e308, 1e308, 0.0])) == 1


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_quantile_order_rejects_invalid_temperatures(temperature: float) -> None:
    """Reject invalid temperatures at both the direct and constructed interfaces.

    Args:
        temperature (float): Invalid numerical temperature.
    """
    with pytest.raises(ValueError, match="temperature"):
        QuantileCodec.order(np.zeros(2), np.arange(2.0), temperature)
    with pytest.raises(ValueError, match="temperature"):
        QuantileDecoder.build(InstanceFactory.make("uncorrelated", 2, 1), temperature)


@settings(max_examples=60, derandomize=True, deadline=None)
@given(
    st.lists(st.floats(min_value=0.0, max_value=np.nextafter(1.0, 0.0)), min_size=1, max_size=20),
    st.floats(min_value=np.nextafter(0.0, 1.0), max_value=1e4),
)
def test_quantile_order_preserves_every_item(keys: list[float], temperature: float) -> None:
    """Every valid key vector and positive finite temperature produces each item once.

    Args:
        keys (list[float]): Generated finite keys in the unit interval.
        temperature (float): Generated positive finite temperature.
    """
    order = QuantileCodec.order(np.asarray(keys), np.linspace(0.0, 1.0, len(keys)), temperature)
    assert sorted(order.tolist()) == list(range(len(keys)))


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
