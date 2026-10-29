"""NumPy network surrogate contract tests.

Objective:
- Check that the network finds structure a linear model cannot, stays silent on noise and is
  deterministic.

Technical approach:
- Use a symmetric quadratic whose linear trend is zero and a pure-noise cost on random keys.

Flow (pseudocode):
1. Sample keys and costs with known structure.
2. Fit the network and a ridge reference on the same samples.
3. Compare held-out correlations and the proposed keys.

Important criteria:
- The network clears the no-signal threshold where the linear fit does not.
- Proposals stay inside [0, 1) and are reproducible for a fixed seed.

Inputs/Outputs:
- Input: sampled keys and costs.
- Output: assertions on correlations and proposed keys.
"""

import numpy as np
from benchmarks.engines.mlp_surrogate import STARTS
from benchmarks.engines.mlp_surrogate import WEIGHT_DECAY
from benchmarks.engines.mlp_surrogate import MlpSurrogate
from benchmarks.engines.surrogate_seed_study import MIN_CORRELATION
from benchmarks.engines.surrogate_seed_study import SurrogateSeeds


def quadratic_samples(count: int, width: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample a symmetric quadratic centered at one half.

    Args:
        count (int): Number of samples.
        width (int): Number of keys.
        seed (int): Sampling seed.

    Returns:
        tuple[np.ndarray, np.ndarray]: Keys and their squared distance to the center.
    """
    keys = np.random.default_rng(seed).random((count, width))
    return keys, np.sum((keys - 0.5) ** 2, axis=1)


def test_network_finds_structure_the_linear_fit_misses() -> None:
    """A symmetric quadratic has no linear trend, yet the network predicts held-out ranks."""
    keys, costs = quadratic_samples(600, 6, 1)
    target = np.argsort(np.argsort(costs)) / (len(costs) - 1) - 0.5
    split = int(0.8 * len(keys))
    centered = keys - 0.5
    linear = SurrogateSeeds.solve(centered[:split], target[:split], 1.0)
    linear_correlation = float(np.corrcoef(centered[split:] @ linear, target[split:])[0, 1])
    network = MlpSurrogate(keys, costs, 3)
    assert abs(linear_correlation) < MIN_CORRELATION
    assert network.fit() > 0.5


def test_proposals_improve_on_random_keys_and_stay_in_range() -> None:
    """Descending the fitted model lands on keys far better than the median random key."""
    keys, costs = quadratic_samples(600, 6, 2)
    seeds = MlpSurrogate(keys, costs, 3).propose(MIN_CORRELATION)
    assert len(seeds) == STARTS
    assert all(0.0 <= key < 1.0 for seed in seeds for key in seed)
    offered = [float(np.sum((np.asarray(seed) - 0.5) ** 2)) for seed in seeds]
    assert max(offered) < float(np.median(costs))


def test_noise_yields_no_proposals() -> None:
    """Costs unrelated to the keys give a held-out correlation below the threshold."""
    keys = np.random.default_rng(4).random((600, 6))
    costs = np.random.default_rng(5).random(600)
    network = MlpSurrogate(keys, costs, 3)
    assert network.propose(MIN_CORRELATION) == []
    assert network.correlation < MIN_CORRELATION


def test_fixed_seed_reproduces_the_proposals() -> None:
    """Initialization is the only randomness, so equal seeds give equal keys."""
    keys, costs = quadratic_samples(300, 5, 6)
    first = MlpSurrogate(keys, costs, 9).propose(MIN_CORRELATION)
    second = MlpSurrogate(keys, costs, 9).propose(MIN_CORRELATION)
    assert first == second and first


def regression_loss(weights: dict[str, np.ndarray], keys: np.ndarray, target: np.ndarray) -> float:
    """Evaluate the regularized loss independently of the derivative implementation.

    Args:
        weights (dict[str, np.ndarray]): Candidate network parameters.
        keys (np.ndarray): Centered input samples.
        target (np.ndarray): Scalar regression targets.

    Returns:
        float: Mean squared residual plus quadratic weight regularization.
    """
    hidden = np.tanh(keys @ weights["w1"] + weights["b1"])
    residual = hidden @ weights["w2"] + weights["b2"] - target
    decay = np.sum(weights["w1"] ** 2) + np.sum(weights["w2"] ** 2)
    return float(np.mean(residual**2) + 0.5 * WEIGHT_DECAY * decay)


def test_parameter_gradients_match_central_finite_differences() -> None:
    """Backpropagation agrees with the independent numerical loss derivative."""
    keys, costs = quadratic_samples(10, 3, 13)
    model = MlpSurrogate(keys, costs, 17)
    weights = model.init(3)
    centered = keys - 0.5
    gradients = model.gradients(weights, centered, model.target)
    step = 1e-6
    for name, value in weights.items():
        numerical = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            original = value[index]
            value[index] = original + step
            above = regression_loss(weights, centered, model.target)
            value[index] = original - step
            below = regression_loss(weights, centered, model.target)
            value[index] = original
            numerical[index] = (above - below) / (2.0 * step)
        np.testing.assert_allclose(gradients[name], numerical, atol=1e-8, rtol=1e-5)
