"""Float64 contextual categorical policies in the isolated neural training environment.

Objective:
- Learn constructive preferences without gradients through hard quantile decisions.

Technical approach:
- Use a small hidden-tanh network, a positive uniform action mixture and detached cost baselines.

Flow (pseudocode):
1. Compute contextual legal-action probabilities.
2. Sample checked trajectories and accumulate their log probabilities.
3. Minimize a score-function cost objective and export frozen numerical parameters.

Important criteria:
- No training-framework imports cross into ordinary solver model evaluation.

Inputs/Outputs:
- Input: validated task adapters, contextual features and local random generators.
- Output: checked rollout costs, differentiable log probabilities and frozen weights.
"""

from __future__ import annotations

import numpy as np
import torch
from benchmarks.quantile.codec import Candidate
from benchmarks.quantile.codec import ConstructionFailure
from benchmarks.quantile.codec import HorizonExhausted
from benchmarks.quantile.codec import TrajectoryCodec
from benchmarks.quantile.model import FrozenScorer
from benchmarks.quantile.probabilities import Intervals
from benchmarks.quantile.tasks import Cover
from benchmarks.quantile.tasks import Tour


class TrainingPolicy:
    """Train the same small network shape evaluated by the frozen numerical scorer.

    Responsibilities:
    - Keep hard decisions and objective values outside the gradient graph.
    - Differentiate the actual mixed categorical policy rather than an unmixed proxy.

    Attributes:
        family (str): Task-feature family.
        parameters (dict[str, torch.Tensor]): Four float64 trainable numerical arrays.
    """

    def __init__(self, family: str, hidden: int = 16, seed: int = 0) -> None:
        """Initialize owned parameters without changing global random generators.

        Args:
            family (str): Supported feature family.
            hidden (int): Hidden width between one and sixty-four.
            seed (int): Local numerical parameter initialization.

        Raises:
            ValueError: If the model configuration is unsupported.
        """
        rng = np.random.default_rng(seed)
        initial = FrozenScorer(
            family,
            rng.normal(0.0, 0.5, (4, hidden)),
            np.zeros(hidden),
            rng.normal(0.0, 1.0 / np.sqrt(hidden), hidden),
            np.array(0.0),
        )
        self.family = family
        self.parameters = {
            name: torch.tensor(
                np.array(getattr(initial, name)), dtype=torch.float64, requires_grad=True
            )
            for name in FrozenScorer.NAMES
        }

    def frozen(self) -> FrozenScorer:
        """Export an immutable snapshot independent of later training updates.

        Returns:
            FrozenScorer: NumPy-only validated frozen model.
        """
        arrays = [self.parameters[name].detach().numpy() for name in FrozenScorer.NAMES]
        return FrozenScorer(self.family, arrays[0], arrays[1], arrays[2], arrays[3])

    def probabilities(self, features: np.ndarray) -> torch.Tensor:
        """Compute the documented clipped, uniform-mixed categorical probabilities.

        Args:
            features (np.ndarray): Finite four-feature legal-action rows.

        Returns:
            torch.Tensor: Positive float64 probabilities with a differentiable score graph.

        Raises:
            ValueError: If features or resulting numerical scores are invalid.
        """
        rows = np.asarray(features, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 4 or not np.isfinite(rows).all():
            raise ValueError("Action features must be a finite four-column matrix.")
        values = torch.tensor(np.array(rows), dtype=torch.float64)
        p = self.parameters
        logits = torch.clamp(
            torch.tanh(values @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"], -10.0, 10.0
        )
        Intervals.validate(logits.detach().numpy(), 1.0, 0.05)
        return 0.95 * torch.softmax(logits, dim=0) + 0.05 / len(logits)

    def log_probability(self, task: Tour | Cover, trace: tuple[int, ...]) -> torch.Tensor:
        """Replay an admissible trajectory and accumulate its actual policy log probability.

        Args:
            task (Tour | Cover): Exact construction semantics.
            trace (tuple[int, ...]): Complete admissible ordered actions.

        Returns:
            torch.Tensor: Differentiable log probability of this trajectory.

        Raises:
            ValueError: If the trace is illegal or incomplete.
        """
        prefix: tuple[int, ...] = ()
        terms = []
        for action in trace:
            actions = task.legal(prefix)
            if action not in actions:
                raise ValueError("The training trace contains a nonlegal action.")
            probabilities = self.probabilities(task.features(prefix, actions))
            terms.append(probabilities[actions.index(action)].log())
            prefix = (*prefix, action)
        task.finish(prefix)
        return torch.stack(terms).sum()

    def rollout(
        self, task: Tour | Cover, rng: np.random.Generator
    ) -> tuple[Candidate, torch.Tensor]:
        """Sample one checked trajectory while retaining only policy log-probability gradients.

        Args:
            task (Tour | Cover): Exact task semantics matching this feature family.
            rng (np.random.Generator): Owned categorical sampling generator.

        Returns:
            tuple[Candidate, torch.Tensor]: Checked trajectory and differentiable log probability.

        Raises:
            ValueError: If the task family or action probabilities are invalid.
            ConstructionFailure: If an unfinished state has no legal actions.
            HorizonExhausted: If the task cannot finish within its declared horizon.
        """
        if (self.family == "tour") != isinstance(task, Tour):
            raise ValueError("Training policy and task family do not match.")
        trace: tuple[int, ...] = ()
        for _ in range(task.horizon):
            if task.terminal(trace):
                break
            actions = task.legal(trace)
            if not actions:
                raise ConstructionFailure("Training encountered a nonterminal empty action set.")
            weights = self.probabilities(task.features(trace, actions)).detach().numpy()
            selected = int(rng.choice(len(actions), p=weights))
            trace = (*trace, actions[selected])
        if not task.terminal(trace):
            raise HorizonExhausted("Training encountered an insufficient task horizon.")
        solution = task.finish(trace)
        return Candidate(solution, trace, task.check(solution)), self.log_probability(task, trace)

    def imitate(
        self, task: Tour | Cover, rng: np.random.Generator, optimizer: torch.optim.Optimizer
    ) -> float:
        """Apply a training-only teacher update from a checked analytic greedy trajectory.

        Args:
            task (Tour | Cover): Training instance, never a validation or test instance.
            rng (np.random.Generator): Owned state-subsampling generator.
            optimizer (torch.optim.Optimizer): Optimizer for this policy's parameters.

        Returns:
            float: Mean negative log probability of eight sampled teacher actions.

        Raises:
            ValueError: If the task or its checked teacher trajectory is invalid.
            RuntimeError: If a training gradient is nonfinite.
        """
        teacher = TrajectoryCodec(task).greedy()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for position in rng.integers(0, len(teacher.trace), size=8):
            prefix = teacher.trace[: int(position)]
            actions = task.legal(prefix)
            probabilities = self.probabilities(task.features(prefix, actions))
            selected = actions.index(teacher.trace[int(position)])
            losses.append(-probabilities[selected].log())
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.parameters.values()), 1.0, error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach())

    def update(
        self,
        tasks: list[Tour | Cover],
        rng: np.random.Generator,
        optimizer: torch.optim.Optimizer,
        rollouts: int = 8,
    ) -> dict[str, float]:
        """Apply one clipped policy-gradient update using checked normalized rollout costs.

        Args:
            tasks (list[Tour | Cover]): Nonempty batch from this training family.
            rng (np.random.Generator): Owned trajectory generator.
            optimizer (torch.optim.Optimizer): Optimizer owning this policy's parameters.
            rollouts (int): At least two independent trajectories per instance.

        Returns:
            dict[str, float]: Normalized mean cost, loss and pre-clipping gradient norm.

        Raises:
            ValueError: If the batch, rollout count or trajectories are invalid.
            RuntimeError: If the gradient norm becomes nonfinite.
        """
        if not tasks or rollouts < 2:
            raise ValueError("A training update needs a nonempty batch and at least two rollouts.")
        optimizer.zero_grad(set_to_none=True)
        losses, means = [], []
        for task in tasks:
            reference = max(TrajectoryCodec(task).greedy().cost, 1e-12)
            samples = [self.rollout(task, rng) for _ in range(rollouts)]
            costs = torch.tensor(
                [candidate.cost / reference for candidate, _ in samples], dtype=torch.float64
            )
            log_probabilities = torch.stack([term for _, term in samples])
            losses.append(self.loss(costs, log_probabilities))
            means.append(float(costs.mean()))
        loss = torch.stack(losses).mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            list(self.parameters.values()), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        return {
            "normalized_cost": float(np.mean(means)),
            "loss": float(loss.detach()),
            "gradient_norm": float(norm),
        }

    @staticmethod
    def loss(costs: torch.Tensor, log_probabilities: torch.Tensor) -> torch.Tensor:
        """Form the minimization score-function loss with detached leave-one-out baselines.

        Args:
            costs (torch.Tensor): Normalized checked rollout costs.
            log_probabilities (torch.Tensor): Matching trajectory log probabilities.

        Returns:
            torch.Tensor: Scalar differentiable policy-gradient loss.

        Raises:
            ValueError: If fewer than two matching finite rollout observations are supplied.
        """
        if costs.ndim != 1 or costs.shape != log_probabilities.shape or len(costs) < 2:
            raise ValueError("The loss requires matching vectors with at least two rollouts.")
        if not torch.isfinite(costs).all() or not torch.isfinite(log_probabilities).all():
            raise ValueError("Rollout costs and log probabilities must be finite.")
        detached = costs.detach()
        baseline = (detached.sum() - detached) / (len(detached) - 1)
        return ((detached - baseline) * log_probabilities).mean()
