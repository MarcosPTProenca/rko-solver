"""Qualify isolated float64 policy gradients and exported numerical scores.

Objective:
- Verify the score-function objective without differentiating hard construction decisions.

Technical approach:
- Enumerate tiny tour trajectories and compare exact gradients with finite differences.

Flow (pseudocode):
1. Compare training probabilities with the frozen numerical implementation.
2. Check detached leave-one-out cost advantages.
3. Differentiate an exactly enumerated expected objective and compare a finite difference.

Important criteria:
- Execute only in the separately locked training environment.

Inputs/Outputs:
- Input: tiny tasks and fixed numerical parameters.
- Output: probability, gradient and safe-export parity assertions.
"""

from __future__ import annotations

import itertools
import numpy as np
import pytest
import torch
from benchmarks.quantile.probabilities import Intervals
from benchmarks.quantile.tasks import Tour
from benchmarks.quantile.training import policy


def test_training_probabilities_match_exported_numerical_scores() -> None:
    """The uniform mixture is identical in training and frozen float64 evaluation."""
    network = policy.TrainingPolicy("tour", hidden=4, seed=7)
    features = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.4, 0.3, 0.2]])
    actual = network.probabilities(features).detach().numpy()
    expected = Intervals.from_logits(network.frozen().scores(features)).probabilities
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)


def test_cost_advantages_are_detached_and_have_the_minimization_sign() -> None:
    """Gradient descent favors the cheaper rollout without differentiating costs."""
    costs = torch.tensor([1.0, 3.0], dtype=torch.float64, requires_grad=True)
    log_probabilities = torch.tensor([-0.2, -1.0], dtype=torch.float64, requires_grad=True)
    loss = policy.TrainingPolicy.loss(costs, log_probabilities)
    loss.backward()
    assert costs.grad is None
    np.testing.assert_array_equal(log_probabilities.grad.numpy(), [-1.0, 1.0])


def expected_cost(network: policy.TrainingPolicy, task: Tour) -> torch.Tensor:
    """Enumerate the actual expected objective over all tiny fixed-start tours.

    Args:
        network (TrainingPolicy): Differentiable contextual policy.
        task (Tour): Tiny complete tour instance.

    Returns:
        torch.Tensor: Exact finite-horizon expected objective with its gradient graph.
    """
    terms = []
    for trace in itertools.permutations(range(1, len(task.coordinates))):
        probability = network.log_probability(task, trace).exp()
        cost = task.check((0, *trace, 0))
        terms.append(probability * cost)
    return torch.stack(terms).sum()


def test_exact_expected_gradient_matches_finite_differences() -> None:
    """The mixture-policy objective has a nontrivial gradient matching an independent difference."""
    network = policy.TrainingPolicy("tour", hidden=4, seed=11)
    task = Tour(np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 1.0], [0.0, 2.0]]))
    expected = expected_cost(network, task)
    gradient = torch.autograd.grad(expected, network.parameters["w2"])[0][0].item()
    parameter = network.parameters["w2"]
    original = parameter[0].item()
    step = 1e-5
    with torch.no_grad():
        parameter[0] = original + step
        above = expected_cost(network, task).item()
        parameter[0] = original - step
        below = expected_cost(network, task).item()
        parameter[0] = original
    assert abs(gradient) > 1e-8
    assert gradient == pytest.approx((above - below) / (2.0 * step), rel=1e-5, abs=1e-8)


def test_checked_rollout_and_update_preserve_frozen_snapshot() -> None:
    """Sampling uses exact costs and later optimization cannot mutate an exported checkpoint."""
    network = policy.TrainingPolicy("tour", hidden=4, seed=11)
    task = Tour(np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 1.0], [0.0, 2.0]]))
    rng = np.random.default_rng(17)
    candidate, log_probability = network.rollout(task, rng)
    assert candidate.cost == task.check(candidate.solution)
    assert log_probability.item() == pytest.approx(
        network.log_probability(task, candidate.trace).item()
    )
    assert log_probability.requires_grad
    frozen = network.frozen()
    fingerprint = frozen.fingerprint
    optimizer = torch.optim.Adam(network.parameters.values(), lr=0.01)
    metrics = network.update([task], rng, optimizer, rollouts=8)
    assert np.isfinite(list(metrics.values())).all()
    assert network.frozen().fingerprint != fingerprint
    assert frozen.fingerprint == fingerprint


def test_teacher_warm_start_uses_only_an_explicit_checked_reference() -> None:
    """A training-only heuristic trace can supply detached supervision before policy gradients."""
    network = policy.TrainingPolicy("tour", hidden=4, seed=11)
    task = Tour(np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 1.0], [0.0, 2.0]]))
    before = network.frozen().fingerprint
    optimizer = torch.optim.Adam(network.parameters.values(), lr=0.01)
    loss = network.imitate(task, np.random.default_rng(17), optimizer)
    assert np.isfinite(loss) and loss >= 0.0
    assert network.frozen().fingerprint != before
