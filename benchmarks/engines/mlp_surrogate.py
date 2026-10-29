"""One-hidden-layer NumPy surrogate that proposes starting keys from sampled evaluations.

Objective:
- Provide a nonlinear counterpart of the ridge surrogate in `surrogate_seed_study`, so the same
  pre-registered test can say whether a small neural model finds structure a linear one cannot.

Technical approach:
- `MlpSurrogate` regresses a centered cost rank on the keys with one tanh hidden layer, trained by
  full-batch Adam with weight decay on 80 percent of the samples. The remaining fifth selects the
  epoch (early stopping) and gives the held-out correlation used as the no-signal test.
- Proposals run projected gradient descent on the fitted model, in key space, from the lowest-cost
  sampled keys; the input gradient is analytic, so the real objective is not evaluated again.
- Everything is NumPy on CPU with a fixed seed; no framework and no GPU.

Flow (pseudocode):
1. Split the samples, standardize nothing beyond centering keys at one half.
2. Train with Adam; every `CHECK_EVERY` epochs score the held-out fifth and keep the best weights.
3. Return no seeds when the best held-out correlation is below the caller's threshold.
4. Otherwise move each start against the model gradient with a fixed step and clip to [0, 1).

Important criteria:
- Training cost is `epochs * samples * n_variables * hidden` multiply-adds; the caller charges the
  elapsed wall time to the solve budget.
- The model never sees the problem, only keys and cost ranks.
- Determinism: the same seed, samples and hyperparameters give identical weights.

Inputs/Outputs:
- Input: sampled key matrix, matching costs and a seed.
- Output: a held-out correlation and, when it clears the threshold, proposed key vectors.
"""

from __future__ import annotations

import numpy as np

HIDDEN = 16
EPOCHS = 150
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-3
CHECK_EVERY = 5
STEPS = 30
STEP_SIZE = 0.02
STARTS = 3
TRAIN_FRACTION = 0.8


class MlpSurrogate:
    """Fit a small tanh network to cost ranks and descend it in key space.

    Responsibilities:
    - Train with early stopping on a held-out fifth and report its correlation.
    - Propose keys by projected gradient descent on the fitted model.

    Attributes:
        keys (np.ndarray): Sampled key matrix.
        target (np.ndarray): Centered cost ranks in [-0.5, 0.5].
        order (np.ndarray): Sample indices sorted by ascending cost.
        correlation (float): Held-out correlation of the selected weights.
        weights (dict[str, np.ndarray] | None): Selected parameters, None before `fit`.
        rng (np.random.Generator): Initialization generator.
    """

    def __init__(self, keys: np.ndarray, costs: np.ndarray, seed: int) -> None:
        """Store the samples and derive the rank target.

        Args:
            keys (np.ndarray): Sampled keys, one row per sample.
            costs (np.ndarray): Cost of each sampled row.
            seed (int): Initialization seed.
        """
        self.keys = keys
        self.order = np.argsort(costs)
        ranks = np.argsort(self.order)
        self.target = ranks / (len(costs) - 1) - 0.5
        self.correlation = -1.0
        self.weights: dict[str, np.ndarray] | None = None
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def forward(
        weights: dict[str, np.ndarray], centered: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the network.

        Args:
            weights (dict[str, np.ndarray]): Parameters `w1`, `b1`, `w2`, `b2`.
            centered (np.ndarray): Keys minus one half.

        Returns:
            tuple[np.ndarray, np.ndarray]: Hidden activations and the scalar outputs.
        """
        hidden = np.tanh(centered @ weights["w1"] + weights["b1"])
        return hidden, hidden @ weights["w2"] + weights["b2"]

    def init(self, width: int) -> dict[str, np.ndarray]:
        """Draw the initial parameters.

        Args:
            width (int): Number of keys.

        Returns:
            dict[str, np.ndarray]: Parameters with Glorot-style scales and zero biases.
        """
        return {
            "w1": self.rng.normal(0.0, 1.0 / np.sqrt(width), (width, HIDDEN)),
            "b1": np.zeros(HIDDEN),
            "w2": self.rng.normal(0.0, 1.0 / np.sqrt(HIDDEN), HIDDEN),
            "b2": np.zeros(()),
        }

    @classmethod
    def gradients(
        cls, weights: dict[str, np.ndarray], centered: np.ndarray, target: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute the mean-squared-error gradients with weight decay.

        Args:
            weights (dict[str, np.ndarray]): Current parameters.
            centered (np.ndarray): Training keys minus one half.
            target (np.ndarray): Training targets.

        Returns:
            dict[str, np.ndarray]: Gradient of every parameter.
        """
        hidden, output = cls.forward(weights, centered)
        residual = (output - target) * (2.0 / len(target))
        back = np.outer(residual, weights["w2"]) * (1.0 - hidden**2)
        return {
            "w1": centered.T @ back + WEIGHT_DECAY * weights["w1"],
            "b1": back.sum(axis=0),
            "w2": hidden.T @ residual + WEIGHT_DECAY * weights["w2"],
            "b2": residual.sum(),
        }

    @staticmethod
    def adam(
        weights: dict[str, np.ndarray],
        grads: dict[str, np.ndarray],
        state: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
        epoch: int,
    ) -> None:
        """Apply one bias-corrected Adam update in place.

        Args:
            weights (dict[str, np.ndarray]): Parameters, updated in place.
            grads (dict[str, np.ndarray]): Gradient of every parameter.
            state (tuple[dict[str, np.ndarray], dict[str, np.ndarray]]): First and second moments,
                updated in place.
            epoch (int): One-based step count for bias correction.
        """
        moment, square = state
        for name, grad in grads.items():
            moment[name] = 0.9 * moment[name] + 0.1 * grad
            square[name] = 0.999 * square[name] + 0.001 * grad * grad
            scaled = (moment[name] / (1 - 0.9**epoch)) / (
                np.sqrt(square[name] / (1 - 0.999**epoch)) + 1e-8
            )
            weights[name] = weights[name] - LEARNING_RATE * scaled

    def fit(self) -> float:
        """Train with Adam and keep the weights with the best held-out correlation.

        Returns:
            float: Held-out correlation of the selected weights.
        """
        centered = self.keys - 0.5
        split = int(TRAIN_FRACTION * len(centered))
        weights = self.init(centered.shape[1])
        state = (
            {name: np.zeros_like(value) for name, value in weights.items()},
            {name: np.zeros_like(value) for name, value in weights.items()},
        )
        for epoch in range(1, EPOCHS + 1):
            grads = self.gradients(weights, centered[:split], self.target[:split])
            self.adam(weights, grads, state, epoch)
            if epoch % CHECK_EVERY == 0:
                self.checkpoint(weights, centered[split:], self.target[split:])
        return self.correlation

    def checkpoint(
        self, weights: dict[str, np.ndarray], centered: np.ndarray, target: np.ndarray
    ) -> None:
        """Keep a copy of the weights when they beat the best held-out correlation so far.

        Args:
            weights (dict[str, np.ndarray]): Current parameters.
            centered (np.ndarray): Held-out keys minus one half.
            target (np.ndarray): Held-out targets.
        """
        held = self.held_out(weights, centered, target)
        if held > self.correlation:
            self.correlation = held
            self.weights = {name: value.copy() for name, value in weights.items()}

    @classmethod
    def held_out(
        cls, weights: dict[str, np.ndarray], centered: np.ndarray, target: np.ndarray
    ) -> float:
        """Correlate predictions with targets on held-out samples.

        Args:
            weights (dict[str, np.ndarray]): Parameters to score.
            centered (np.ndarray): Held-out keys minus one half.
            target (np.ndarray): Held-out targets.

        Returns:
            float: Pearson correlation, or -1 when the predictions are constant.
        """
        predicted = cls.forward(weights, centered)[1]
        if np.std(predicted) == 0.0:
            return -1.0
        return float(np.corrcoef(predicted, target)[0, 1])

    def descend(self, start: np.ndarray) -> np.ndarray:
        """Move one key vector against the fitted model gradient.

        Args:
            start (np.ndarray): Key vector to start from.

        Returns:
            np.ndarray: Key vector after `STEPS` normalized, clipped gradient steps.
        """
        assert self.weights is not None
        point = np.array(start, dtype=np.float64)
        for _ in range(STEPS):
            hidden, _ = self.forward(self.weights, point - 0.5)
            grad = ((1.0 - hidden**2) * self.weights["w2"]) @ self.weights["w1"].T
            point = np.clip(
                point - STEP_SIZE * grad / (np.max(np.abs(grad)) + 1e-12), 0.0, 0.999999
            )
        return point

    def propose(self, minimum_correlation: float) -> list[list[float]]:
        """Fit and propose keys, or stay silent when the model found no signal.

        Args:
            minimum_correlation (float): Held-out correlation below which no seed is offered.

        Returns:
            list[list[float]]: One descended key vector per start, empty without signal.
        """
        if self.fit() < minimum_correlation or self.weights is None:
            return []
        starts = self.keys[self.order[:STARTS]]
        return [self.descend(start).tolist() for start in starts]
