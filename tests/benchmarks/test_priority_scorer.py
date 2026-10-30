"""Priority scorer and policy-gradient tests.

Objective:
- Verify the gradient, the sampler, the baseline and that training learns a known rule.

Technical approach:
- Compare analytic gradients with central finite differences, check sampler frequencies against
  the exact Plackett-Luce probabilities, and train on a family whose best rule is known.

Flow (pseudocode):
1. Differentiate log-probabilities and network outputs numerically and analytically.
2. Sample orders and compare with exact probabilities.
3. Train briefly and require a clear improvement over the untrained scorer.

Important criteria:
- Gradients agree to numerical precision.
- Training must improve held-out reward and reproduce for a fixed seed.

Inputs/Outputs:
- Input: random scores, small instances and fixed seeds.
- Output: assertions on gradients, probabilities, advantages and rewards.
"""

import itertools
import numpy as np
import pytest
from benchmarks.engines.packing_problems import PackingFamily
from benchmarks.engines.packing_problems import PackingInstance
from benchmarks.engines.priority_scorer import PolicyGradient
from benchmarks.engines.priority_scorer import PriorityScorer
from benchmarks.engines.priority_scorer import ReinforceTrainer
from collections.abc import Callable
from pathlib import Path


def numerical(
    function: Callable[[np.ndarray], float], point: np.ndarray, step: float = 1e-6
) -> np.ndarray:
    """Differentiate a scalar function of a vector by central differences.

    Args:
        function (Callable[[np.ndarray], float]): Scalar function of a vector.
        point (np.ndarray): Evaluation point.
        step (float): Finite-difference step.

    Returns:
        np.ndarray: Numerical gradient.
    """
    gradient = np.zeros_like(point)
    for index in range(len(point)):
        high, low = point.copy(), point.copy()
        high[index] += step
        low[index] -= step
        gradient[index] = (function(high) - function(low)) / (2.0 * step)
    return gradient


def test_log_prob_gradient_matches_finite_differences() -> None:
    """The suffix-normalizer formula equals the numerical derivative of the log-probability."""
    scores = np.random.default_rng(1).normal(size=9)
    order = np.random.default_rng(2).permutation(9)
    analytic = PolicyGradient.log_prob_gradient(scores, order)
    expected = numerical(lambda s: PolicyGradient.log_prob(s, order), scores)
    np.testing.assert_allclose(analytic, expected, atol=1e-7)


def test_order_probabilities_sum_to_one_and_sampler_matches_them() -> None:
    """Exact probabilities of all orders sum to one, and sampled frequencies agree."""
    scores = np.array([0.8, -0.3, 0.1, 0.5])
    orders = [np.array(p) for p in itertools.permutations(range(4))]
    probabilities = np.array([np.exp(PolicyGradient.log_prob(scores, o)) for o in orders])
    assert probabilities.sum() == pytest.approx(1.0)
    drawn = PolicyGradient.sample_orders(scores, 40_000, np.random.default_rng(3))
    counts = {tuple(o): 0 for o in orders}
    for row in drawn:
        counts[tuple(row)] += 1
    empirical = np.array([counts[tuple(o)] for o in orders]) / len(drawn)
    np.testing.assert_allclose(empirical, probabilities, atol=0.01)
    assert all(sorted(row) == [0, 1, 2, 3] for row in drawn[:50].tolist())


def test_leave_one_out_advantages_sum_to_zero_and_match_the_definition() -> None:
    """Each advantage is the reward minus the mean of the others."""
    rewards = np.array([1.0, 2.0, 4.0, 3.0])
    advantages = PolicyGradient.advantages(rewards)
    assert advantages.sum() == pytest.approx(0.0)
    for index in range(4):
        others = np.delete(rewards, index)
        assert advantages[index] == pytest.approx(rewards[index] - others.mean())
    with pytest.raises(ValueError, match="two"):
        PolicyGradient.advantages(np.array([1.0]))


def test_network_gradients_match_finite_differences() -> None:
    """Back-propagation through the scorer equals the numerical derivative of a linear loss."""
    features = np.random.default_rng(4).normal(size=(6, 4))
    upstream = np.random.default_rng(5).normal(size=6)
    scorer = PriorityScorer.init(4, 5, 6)
    analytic = scorer.gradients(features, upstream)
    for name, value in scorer.weights.items():
        flat = value.reshape(-1).copy()

        def loss(
            vector: np.ndarray, name: str = name, shape: tuple[int, ...] = value.shape
        ) -> float:
            """Evaluate the linear loss with one parameter block replaced.

            Args:
                vector (np.ndarray): Flattened replacement block.
                name (str): Parameter being varied.
                shape (tuple[int, ...]): Original block shape.

            Returns:
                float: Upstream-weighted sum of scores.
            """
            trial = PriorityScorer({k: v.copy() for k, v in scorer.weights.items()})
            trial.weights[name] = vector.reshape(shape)
            return float(upstream @ trial.scores(features))

        np.testing.assert_allclose(
            analytic[name].reshape(-1), numerical(loss, flat), atol=1e-6, err_msg=name
        )


def test_save_and_load_round_trip_exactly(tmp_path: Path) -> None:
    """A reloaded scorer reproduces every score bit for bit.

    Args:
        tmp_path (Path): Pytest-provided directory.
    """
    features = np.random.default_rng(7).normal(size=(10, 4))
    scorer = PriorityScorer.init(4, 6, 8)
    scorer.save(tmp_path / "scorer.npz")
    loaded = PriorityScorer.load(tmp_path / "scorer.npz")
    np.testing.assert_array_equal(loaded.scores(features), scorer.scores(features))


def uncorrelated(seed: int) -> PackingInstance:
    """Generate a one-constraint instance whose best rule is profit over weight.

    Args:
        seed (int): Instance seed.

    Returns:
        PackingInstance: Instance with independent profits and weights.
    """
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 101, (1, 30)).astype(np.float64)
    profits = rng.integers(1, 101, 30).astype(np.float64)
    return PackingInstance(profits, weights, np.array([weights.sum() // 2]), f"unc-{seed}")


def test_training_improves_held_out_reward_and_is_reproducible() -> None:
    """On uncorrelated instances an untrained scorer is poor and 150 steps fix most of it."""
    held_out = [uncorrelated(900 + k) for k in range(8)]
    results = []
    for _ in range(2):
        trainer = ReinforceTrainer(PriorityScorer.init(3, 8, 0), 8, 0)
        before = trainer.evaluate(held_out)
        history = trainer.train(uncorrelated, range(100, 164), 150, 8)
        results.append((before, trainer.evaluate(held_out), history))
    before, after, history = results[0]
    assert after > before + 0.05 and after > 0.9
    assert len(history) == 150
    assert results[0][1] == results[1][1] and results[0][2] == results[1][2]


def test_reward_is_normalized_by_the_hand_prior_fill() -> None:
    """The hand-prior order itself earns reward one."""
    instance = PackingFamily.make("mkp", 30, 3)
    trainer = ReinforceTrainer(PriorityScorer.init(7, 4, 1), 4, 1)
    hand_order = np.argsort(-PackingFamily.hand_scores(instance), kind="stable")
    assert trainer.reward(instance, hand_order) == pytest.approx(1.0)
