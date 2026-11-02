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
- `RewardCache` normalizes an order's raw objective by a cached per-instance reference value, so
  one scorer-independent reward definition serves every problem family.
- `ReinforceTrainer` ascends the leave-one-out-baselined score-function estimate of the expected
  reward returned by a caller-supplied features function and reward function, with Adam. Neither
  function is specific to Knapsack; a caller plugs in its own problem.

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
- Input: a features function, a reward function, instances and training settings.
- Output: a fitted scorer, a reward history and an `.npz` artifact.
"""

from __future__ import annotations

import numpy as np
import time
from benchmarks.engines.mlp_surrogate import MlpSurrogate
from benchmarks.engines.report_sink import require_disjoint
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Protocol
from typing import TypeVar


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


@dataclass(slots=True)
class ReinforceStudyDesign:
    """Fixed numerical design shared by every REINFORCE-scorer study.

    Responsibilities:
    - Keep instance size, budgets and the disjoint training, validation and test seed ranges in
      one validated place; a subclass redeclares the defaults that suit its problem and size.

    Attributes:
        n_items (int): Items or cities per instance.
        budgets (tuple[float, ...]): Per-instance wall allowances in seconds.
        test_instances (int): Held-out instances per family.
        validation_instances (int): Instances used to choose spreads.
        train_seeds (tuple[int, int]): Half-open range of training instance seeds.
        validation_base (int): First validation seed.
        test_base (int): First test seed.
        steps (int): REINFORCE updates.
        batch (int): Instances per update.
        samples (int): Sampled orders per instance.
        hidden (int): Scorer hidden width.
        execution (str): Python-engine execution mode passed to the solver.
        algorithms (tuple[str, ...] | None): Portfolio override; None keeps the default portfolio.
    """

    n_items: int = 250
    budgets: tuple[float, ...] = (2.0, 10.0)
    test_instances: int = 12
    validation_instances: int = 6
    train_seeds: tuple[int, int] = (100_000, 101_000)
    validation_base: int = 6000
    test_base: int = 7000
    steps: int = 500
    batch: int = 8
    samples: int = 16
    hidden: int = 16
    execution: str = "process"
    algorithms: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Require pairwise disjoint training, validation and test seeds.

        Raises:
            ValueError: If any two seed ranges overlap.
        """
        require_disjoint(
            (
                range(*self.train_seeds),
                range(self.validation_base, self.validation_base + self.validation_instances),
                range(self.test_base, self.test_base + self.test_instances),
            ),
            "Training, validation and test seeds must be disjoint.",
        )


class NamedInstance(Protocol):
    """An instance that carries a `<family>-<size>-<seed>` name, for reward caching."""

    @property
    def name(self) -> str:
        """Return the instance name.

        Returns:
            str: `<family>-<size>-<seed>`.
        """
        ...


InstanceT = TypeVar("InstanceT", bound=NamedInstance)


class RewardCache:
    """Normalize an order's objective by a cached per-instance reference order's objective.

    Responsibilities:
    - Evaluate a caller-supplied reward function once per instance for a fixed reference order,
      so every sampled order of that instance is compared on the same scale.

    Attributes:
        reward (Callable[[InstanceT, np.ndarray], float]): Raw, unnormalized objective of one order
            on one instance; larger is better.
        reference (Callable[[InstanceT], np.ndarray]): Reference order for one instance, for
            example a hand-written prior's order.
        cache (dict[str, float]): Reference objective per instance name, filled lazily.
    """

    def __init__(
        self,
        reward: Callable[[InstanceT, np.ndarray], float],
        reference: Callable[[InstanceT], np.ndarray],
    ) -> None:
        """Store the raw reward and reference-order functions.

        Args:
            reward (Callable[[InstanceT, np.ndarray], float]): Raw objective of one order.
            reference (Callable[[InstanceT], np.ndarray]): Reference order for one instance.
        """
        self.reward = reward
        self.reference = reference
        self.cache: dict[str, float] = {}

    def __call__(self, instance: InstanceT, order: np.ndarray) -> float:
        """Return the order's objective normalized by the instance's reference objective.

        Args:
            instance (InstanceT): Instance.
            order (np.ndarray): Item order, best-first.

        Returns:
            float: Normalized objective.
        """
        if instance.name not in self.cache:
            self.cache[instance.name] = self.reward(instance, self.reference(instance))
        return self.reward(instance, order) / self.cache[instance.name]


class ReinforceTrainer:
    """Train a scorer with REINFORCE and a leave-one-out baseline, on any problem family.

    Responsibilities:
    - Estimate the policy gradient over a batch of instances and apply Adam updates.
    - Run a full training loop on instances drawn from a seed range.

    Attributes:
        scorer (PriorityScorer): Scorer being trained.
        features (Callable[[InstanceT], np.ndarray]): Per-item feature rows for one instance.
        reward (Callable[[InstanceT, np.ndarray], float]): Normalized reward of one order.
        samples (int): Orders sampled per instance.
        rng (np.random.Generator): Sampling and batching generator.
    """

    def __init__(
        self,
        scorer: PriorityScorer,
        features: Callable[[InstanceT], np.ndarray],
        reward: Callable[[InstanceT, np.ndarray], float],
        samples: int,
        seed: int,
    ) -> None:
        """Store the scorer, the problem-specific functions and initialize optimizer state.

        Args:
            scorer (PriorityScorer): Scorer to train.
            features (Callable[[InstanceT], np.ndarray]): Per-item feature rows for one instance.
            reward (Callable[[InstanceT, np.ndarray], float]): Normalized reward of one order.
            samples (int): Orders sampled per instance, at least two.
            seed (int): Generator seed.

        Raises:
            ValueError: If `samples` is below two.
        """
        if samples < 2:
            raise ValueError("'samples' must be at least two.")
        self.scorer = scorer
        self.features = features
        self.reward = reward
        self.samples = samples
        self.rng = np.random.default_rng(seed)
        self.moments = (
            {name: np.zeros_like(value) for name, value in scorer.weights.items()},
            {name: np.zeros_like(value) for name, value in scorer.weights.items()},
        )
        self.updates = 0

    def step(self, batch: list[InstanceT]) -> float:
        """Apply one policy-gradient update.

        Args:
            batch (list[InstanceT]): Training instances.

        Returns:
            float: Mean sampled reward before the update.
        """
        total = {name: np.zeros_like(value) for name, value in self.scorer.weights.items()}
        rewards = []
        for instance in batch:
            features = self.features(instance)
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
        self, make: Callable[[int], InstanceT], seeds: range, steps: int, batch: int
    ) -> list[float]:
        """Train for a number of steps on instances drawn from a seed range.

        Args:
            make (Callable[[int], InstanceT]): Instance factory taking a seed.
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

    def evaluate(self, instances: list[InstanceT]) -> float:
        """Return the mean normalized reward of the scorer's deterministic order.

        Args:
            instances (list[InstanceT]): Held-out instances.

        Returns:
            float: Mean normalized reward of the descending-score order.
        """
        rewards = []
        for instance in instances:
            scores = self.scorer.scores(self.features(instance))
            rewards.append(self.reward(instance, np.argsort(-scores, kind="stable")))
        return float(np.mean(rewards))

    def fit(
        self,
        kind: str,
        make: Callable[[int], InstanceT],
        held: list[InstanceT],
        seeds: range,
        schedule: tuple[int, int],
    ) -> dict[str, Any]:
        """Evaluate, train for the given step count and report the learning curve.

        Args:
            kind (str): Instance family, recorded in the report only.
            make (Callable[[int], InstanceT]): Instance factory taking a seed.
            held (list[InstanceT]): Held-out instances for the before/after evaluation.
            seeds (range): Training seeds, disjoint from validation and test seeds.
            schedule (tuple[int, int]): Number of updates and instances per update.

        Returns:
            dict[str, Any]: `kind`, training seconds, reward before and after training, the
            mean reward of the first and last ten steps, and the step count.
        """
        steps, batch = schedule
        before = self.evaluate(held)
        started = time.perf_counter()
        history = self.train(make, seeds, steps, batch)
        return {
            "kind": kind,
            "seconds": time.perf_counter() - started,
            "validation_reward_before": before,
            "validation_reward_after": self.evaluate(held),
            "reward_first_10": float(np.mean(history[:10])),
            "reward_last_10": float(np.mean(history[-10:])),
            "steps": steps,
        }
