"""Per-item priority scorer trained with REINFORCE on generated instances.

Objective:
- Learn, offline and per problem family, an item-priority function from instance features alone,
  to replace a hand-written ratio rule inside a residual-priority decoder.

Technical approach:
- `PriorityScorer` is a one-hidden-layer tanh network mapping an item's features to a scalar
  score; inference is NumPy only and the weights serialize to a small `.npz` file.
- `PolicyGradient` treats the scores as a Plackett-Luce distribution over item orders. Orders are
  sampled with Gumbel perturbations; the exact log-probability gradient with respect to the
  scores uses suffix normalizers in O(n log n) per order.
- `ReinforceTrainer` ascends the leave-one-out-baselined score-function estimate of the expected
  normalized profit of the greedy fill, with Adam.

Flow (pseudocode):
1. Draw a batch of generated instances from training seeds.
2. Sample K orders per instance, fill greedily and normalize the profit by a fixed reference.
3. Subtract the leave-one-out baseline and accumulate the score gradient.
4. Back-propagate into the network and apply an Adam step.

Important criteria:
- Training and validation seeds are disjoint from test seeds, enforced by the caller.
- The gradient is checked against finite differences in the tests.
- Training is offline and its cost is reported separately from per-instance inference.

Inputs/Outputs:
- Input: item features, instances and training settings.
- Output: a fitted scorer, a reward history and an `.npz` artifact.
"""

from __future__ import annotations

import numpy as np
from benchmarks.engines.mlp_surrogate import MlpSurrogate
from benchmarks.engines.packing_problems import PackingFamily
from benchmarks.engines.packing_problems import PackingInstance
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, eq=False)
class PriorityScorer:
    """One-hidden-layer tanh network scoring items from their features.

    Responsibilities:
    - Evaluate scores and back-propagate a score gradient.
    - Save and load its parameters.

    Attributes:
        weights (dict[str, np.ndarray]): Parameters `w1`, `b1`, `w2`, `b2`.
    """

    weights: dict[str, np.ndarray]

    @classmethod
    def init(cls, n_features: int, hidden: int, seed: int) -> PriorityScorer:
        """Draw small random parameters.

        Args:
            n_features (int): Number of item features.
            hidden (int): Hidden width.
            seed (int): Initialization seed.

        Returns:
            PriorityScorer: Untrained scorer.
        """
        rng = np.random.default_rng(seed)
        return cls(
            {
                "w1": rng.normal(0.0, 1.0 / np.sqrt(n_features), (n_features, hidden)),
                "b1": np.zeros(hidden),
                "w2": rng.normal(0.0, 1.0 / np.sqrt(hidden), hidden),
                "b2": np.zeros(()),
            }
        )

    def scores(self, features: np.ndarray) -> np.ndarray:
        """Score every item.

        Args:
            features (np.ndarray): One feature row per item.

        Returns:
            np.ndarray: Score per item.
        """
        hidden = np.tanh(features @ self.weights["w1"] + self.weights["b1"])
        return np.asarray(hidden @ self.weights["w2"] + self.weights["b2"])

    def gradients(self, features: np.ndarray, grad_scores: np.ndarray) -> dict[str, np.ndarray]:
        """Back-propagate a gradient with respect to the scores.

        Args:
            features (np.ndarray): One feature row per item.
            grad_scores (np.ndarray): Gradient of a scalar objective with respect to each score.

        Returns:
            dict[str, np.ndarray]: Gradient of every parameter.
        """
        hidden = np.tanh(features @ self.weights["w1"] + self.weights["b1"])
        back = np.outer(grad_scores, self.weights["w2"]) * (1.0 - hidden**2)
        return {
            "w1": features.T @ back,
            "b1": back.sum(axis=0),
            "w2": hidden.T @ grad_scores,
            "b2": np.asarray(grad_scores.sum()),
        }

    def save(self, path: Path) -> None:
        """Write the parameters to an `.npz` file.

        Args:
            path (Path): Destination file.
        """
        np.savez(path, **self.weights)

    @classmethod
    def load(cls, path: Path) -> PriorityScorer:
        """Read parameters written by `save`.

        Args:
            path (Path): Source file.

        Returns:
            PriorityScorer: Scorer with identical parameters.
        """
        with np.load(path) as stored:
            return cls({name: stored[name].copy() for name in stored.files})


class PolicyGradient:
    """Plackett-Luce sampling, log-probability gradient and leave-one-out advantages.

    Responsibilities:
    - Sample orders from a score vector and differentiate their log-probability.
    - Turn rewards into leave-one-out advantages.
    """

    @staticmethod
    def sample_orders(scores: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
        """Sample orders by Gumbel-perturbed sorting.

        Args:
            scores (np.ndarray): Score per item.
            count (int): Number of orders.
            rng (np.random.Generator): Sampling generator.

        Returns:
            np.ndarray: Matrix with one sampled order per row, best-first.
        """
        perturbed = np.asarray(scores) + rng.gumbel(size=(count, len(scores)))
        return np.argsort(-perturbed, axis=1)

    @staticmethod
    def log_prob(scores: np.ndarray, order: np.ndarray) -> float:
        """Return the log-probability of an order under the Plackett-Luce model.

        Args:
            scores (np.ndarray): Score per item.
            order (np.ndarray): Item indices, best-first.

        Returns:
            float: Log-probability.
        """
        ordered = np.asarray(scores)[order]
        shift = ordered.max()
        suffix = np.cumsum(np.exp(ordered - shift)[::-1])[::-1]
        return float(np.sum(ordered - shift - np.log(suffix)))

    @staticmethod
    def log_prob_gradient(scores: np.ndarray, order: np.ndarray) -> np.ndarray:
        """Differentiate the log-probability with respect to the scores.

        Args:
            scores (np.ndarray): Score per item.
            order (np.ndarray): Item indices, best-first.

        Returns:
            np.ndarray: Gradient per item.
        """
        ordered = np.asarray(scores)[order]
        weight = np.exp(ordered - ordered.max())
        suffix = np.cumsum(weight[::-1])[::-1]
        gradient = np.empty(len(order))
        gradient[order] = 1.0 - weight * np.cumsum(1.0 / suffix)
        return gradient

    @staticmethod
    def advantages(rewards: np.ndarray) -> np.ndarray:
        """Subtract each sample's leave-one-out baseline.

        Args:
            rewards (np.ndarray): At least two rewards for one instance.

        Returns:
            np.ndarray: Advantage per sample.

        Raises:
            ValueError: If fewer than two rewards are given.
        """
        if len(rewards) < 2:
            raise ValueError("At least two rewards are required for a leave-one-out baseline.")
        return np.asarray(rewards - (rewards.sum() - rewards) / (len(rewards) - 1))


class ReinforceTrainer:
    """Train a scorer with REINFORCE and a leave-one-out baseline.

    Responsibilities:
    - Estimate the policy gradient over a batch of instances and apply Adam updates.
    - Run a full training loop on instances drawn from a seed range.

    Attributes:
        scorer (PriorityScorer): Scorer being trained.
        samples (int): Orders sampled per instance.
        rng (np.random.Generator): Sampling and batching generator.
    """

    def __init__(self, scorer: PriorityScorer, samples: int, seed: int) -> None:
        """Store the scorer and initialize optimizer state.

        Args:
            scorer (PriorityScorer): Scorer to train.
            samples (int): Orders sampled per instance, at least two.
            seed (int): Generator seed.
        """
        if samples < 2:
            raise ValueError("'samples' must be at least two.")
        self.scorer = scorer
        self.samples = samples
        self.rng = np.random.default_rng(seed)
        self.moments = (
            {name: np.zeros_like(value) for name, value in scorer.weights.items()},
            {name: np.zeros_like(value) for name, value in scorer.weights.items()},
        )
        self.updates = 0
        self.reference: dict[str, float] = {}

    def reward(self, instance: PackingInstance, order: np.ndarray) -> float:
        """Return the greedy-fill profit normalized by the hand-prior fill profit.

        Args:
            instance (PackingInstance): Packing instance.
            order (np.ndarray): Item order, best-first.

        Returns:
            float: Normalized profit.
        """
        if instance.name not in self.reference:
            hand = np.argsort(-PackingFamily.hand_scores(instance), kind="stable")
            self.reference[instance.name] = PackingFamily.fill(hand, instance)[1]
        return PackingFamily.fill(order, instance)[1] / self.reference[instance.name]

    def step(self, batch: list[PackingInstance]) -> float:
        """Apply one policy-gradient update.

        Args:
            batch (list[PackingInstance]): Training instances.

        Returns:
            float: Mean sampled reward before the update.
        """
        total = {name: np.zeros_like(value) for name, value in self.scorer.weights.items()}
        rewards = []
        for instance in batch:
            features = PackingFamily.features(instance)
            scores = self.scorer.scores(features)
            orders = PolicyGradient.sample_orders(scores, self.samples, self.rng)
            sampled = np.array([self.reward(instance, order) for order in orders])
            advantage = PolicyGradient.advantages(sampled)
            direction = sum(
                a * PolicyGradient.log_prob_gradient(scores, order)
                for a, order in zip(advantage, orders, strict=True)
            )
            for name, grad in self.scorer.gradients(features, direction / self.samples).items():
                total[name] += grad
            rewards.append(float(sampled.mean()))
        self.updates += 1
        ascent = {name: -grad / len(batch) for name, grad in total.items()}
        MlpSurrogate.adam(self.scorer.weights, ascent, self.moments, self.updates)
        return float(np.mean(rewards))

    def train(
        self, make: Callable[[int], PackingInstance], seeds: range, steps: int, batch: int
    ) -> list[float]:
        """Train for a number of steps on instances drawn from a seed range.

        Args:
            make (Callable[[int], PackingInstance]): Instance factory taking a seed.
            seeds (range): Training seeds, disjoint from validation and test seeds.
            steps (int): Number of updates.
            batch (int): Instances per update.

        Returns:
            list[float]: Mean sampled reward at every step.
        """
        history = []
        for _ in range(steps):
            chosen = self.rng.choice(np.arange(seeds.start, seeds.stop), size=batch)
            history.append(self.step([make(int(seed)) for seed in chosen]))
        return history

    def evaluate(self, instances: list[PackingInstance]) -> float:
        """Return the mean normalized profit of the scorer's deterministic order.

        Args:
            instances (list[PackingInstance]): Held-out instances.

        Returns:
            float: Mean normalized profit of the descending-score fill.
        """
        rewards = []
        for instance in instances:
            scores = self.scorer.scores(PackingFamily.features(instance))
            rewards.append(self.reward(instance, np.argsort(-scores, kind="stable")))
        return float(np.mean(rewards))
