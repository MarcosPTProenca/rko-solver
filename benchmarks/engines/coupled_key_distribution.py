"""Full-dimensional conditional key distributions for decoder-adapter screening.

Objective:
- Preserve every latent search direction while learning dependencies between elite keys.

Technical approach:
- Convert uniform latent keys to standardized logistic coordinates, then apply marginal location
  and scale estimates shrunk toward the uniform-key prior.
- Train a small tanh network to predict centered odd-coordinate logits from even coordinates.
  Add only a bounded conditional shift to the odd block; the even block remains unchanged.
- Both blocks retain their own random coordinates. The triangular shift has an explicit inverse
  in the numerical interior, unlike a low-dimensional bottleneck generator.
- Keep the neural component only if its validation reconstruction error improves over the
  zero-shift baseline by at least ten percent. This validates reconstruction, not optimization gain.

Flow (pseudocode):
1. Fit shrunk marginal statistics from the elite archive.
2. Train and validate the conditional network within its fixed allowance.
3. Generate all coordinates, then shift the odd block conditionally on the even block.
4. Compose the generated keys with the caller's unchanged decoder.

Important criteria:
- This is a benchmark prototype, not an installed API.
- No problem features or objective predictions enter the model.
- Floating-point clipping limits exact inverse claims to the interior of the key cube.

Inputs/Outputs:
- Input: elite keys and latent random keys.
- Output: full-dimensional generated keys and their inverse representation.
"""

from __future__ import annotations

import numpy as np
import time
from benchmarks.engines.mlp_surrogate import MlpSurrogate
from benchmarks.engines.neural_key_adapter import KeyDistribution

LOGISTIC_SCALE = float(np.pi / np.sqrt(3.0))
PRIOR_WEIGHT = 0.75
SHIFT_SCALE = 0.25
VALIDATION_RATIO = 0.9


class CoupledDistribution(KeyDistribution):
    """Learn a full-dimensional conditional key transform.

    Responsibilities:
    - Preserve every latent coordinate rather than compressing the search space.
    - Provide an inverse on interior unit-cube values.
    """

    @classmethod
    def marginal(cls, elite: np.ndarray) -> CoupledDistribution:
        """Shrink empirical logit moments toward the uniform-key prior.

        Args:
            elite (np.ndarray): Evaluated elite keys.

        Returns:
            CoupledDistribution: Full-dimensional independent-coordinate control.

        Raises:
            ValueError: If the archive is invalid.
        """
        logits = cls.coordinates(elite)
        mean = (1.0 - PRIOR_WEIGHT) * logits.mean(axis=0)
        variance = (1.0 - PRIOR_WEIGHT) * logits.var(axis=0) + PRIOR_WEIGHT * LOGISTIC_SCALE**2
        return cls(mean, np.sqrt(variance))

    @classmethod
    def fit(
        cls, elite: np.ndarray, seed: int, epochs: int = 40, deadline: float | None = None
    ) -> CoupledDistribution:
        """Fit and validate a bounded conditional neural shift from elite keys.

        Args:
            elite (np.ndarray): Evaluated elite keys.
            seed (int): Training seed.
            epochs (int): Positive training-step cap.
            deadline (float | None): Optional monotonic training deadline.

        Returns:
            CoupledDistribution: Neural generator, or the non-neural control without signal.

        Raises:
            ValueError: If the archive or epoch count is invalid.
        """
        if epochs < 1:
            raise ValueError("'epochs' must be positive.")
        model = cls.marginal(elite)
        logits = cls.coordinates(elite)
        features = (logits[:, ::2] - model.mean[::2]) / model.scale[::2]
        target = (logits[:, 1::2] - logits[:, 1::2].mean(axis=0)) / model.scale[1::2]
        if target.shape[1]:
            model.optimize((features, target), seed, epochs, deadline)
        return model

    def optimize(
        self, data: tuple[np.ndarray, np.ndarray], seed: int, epochs: int, deadline: float | None
    ) -> None:
        """Train the conditional reconstruction network with an epoch-level deadline.

        Args:
            data (tuple[np.ndarray, np.ndarray]): Standardized features and centered targets.
            seed (int): Initialization and corruption seed.
            epochs (int): Maximum training steps.
            deadline (float | None): Optional monotonic deadline.
        """
        features, target = data
        split = max(1, int(0.8 * len(features)))
        rng = np.random.default_rng(seed)
        width = min(8, features.shape[1])
        weights, state = self.initialize(features.shape[1], width, target.shape[1], rng)
        for epoch in range(1, epochs + 1):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            noisy = features[:split] + rng.normal(0.0, 0.05, features[:split].shape)
            grads = self.gradients(weights, noisy, target[:split])
            MlpSurrogate.adam(weights, grads, state, epoch)
            self.epochs = epoch
        self.validate(weights, features[split:], target[split:])

    def validate(
        self, weights: dict[str, np.ndarray], features: np.ndarray, target: np.ndarray
    ) -> None:
        """Activate only a completed network that improves held-out reconstruction.

        Args:
            weights (dict[str, np.ndarray]): Conditional reconstruction parameters.
            features (np.ndarray): Validation inputs.
            target (np.ndarray): Validation targets.
        """
        hidden = np.tanh(features @ weights["w1"] + weights["b1"])
        prediction = hidden @ weights["w2"] + weights["b2"]
        self.error = float(np.mean((prediction - target) ** 2) / max(np.mean(target**2), 1e-12))
        if self.epochs and self.error < VALIDATION_RATIO:
            self.weights = weights

    def shift(self, logits: np.ndarray) -> np.ndarray:
        """Compute the bounded odd-coordinate shift conditioned on the even block.

        Args:
            logits (np.ndarray): Original-space logits with the even block already generated.

        Returns:
            np.ndarray: Additive shifts for the odd coordinates.
        """
        if self.weights is None:
            return np.zeros(len(logits[1::2]))
        inputs = (logits[::2] - self.mean[::2]) / self.scale[::2]
        hidden = np.tanh(inputs @ self.weights["w1"] + self.weights["b1"])
        prediction = hidden @ self.weights["w2"] + self.weights["b2"]
        return SHIFT_SCALE * self.scale[1::2] * np.tanh(prediction)

    @staticmethod
    def logits(keys: np.ndarray) -> np.ndarray:
        """Convert interior keys to logits with explicit endpoint stabilization.

        Args:
            keys (np.ndarray): Random-key vector.

        Returns:
            np.ndarray: Numerically bounded logit vector.
        """
        bounded = np.clip(keys, 1e-10, 1.0 - 1e-10)
        return np.log(bounded) - np.log1p(-bounded)

    def transform(self, keys: np.ndarray) -> np.ndarray:
        """Generate all key coordinates and apply the conditional triangular shift.

        Args:
            keys (np.ndarray): Latent random keys.

        Returns:
            np.ndarray: Generated original-space keys.
        """
        logits = self.mean + self.scale * self.logits(np.asarray(keys)) / LOGISTIC_SCALE
        logits[1::2] += self.shift(logits)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

    def inverse(self, keys: np.ndarray) -> np.ndarray:
        """Recover the latent representation of interior generated keys.

        Args:
            keys (np.ndarray): Generated original-space keys.

        Returns:
            np.ndarray: Corresponding latent vector, subject to floating-point endpoint limits.
        """
        logits = self.logits(np.asarray(keys))
        logits[1::2] -= self.shift(logits)
        latent = LOGISTIC_SCALE * (logits - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-np.clip(latent, -20.0, 20.0)))
