"""Learned key distributions composed with an existing problem decoder.

Objective:
- Test a neural generator rather than a cost predictor without using domain features.

Technical approach:
- Transform an elite archive into standardized logit coordinates and fit a small denoising
  autoencoder: a tanh encoder and a linear decoder, trained with full-batch Adam.
- During search, use latent random keys to drive the learned decoder, undo the standardization
  and apply the sigmoid. This generates keys for the caller's unchanged problem decoder.
- Retain a protected uniform branch. Rescaling its first coordinate preserves the full original
  key cube; existing solutions can be encoded exactly through that branch.
- The non-neural control uses the same archive mean and coordinate scales without learned
  correlations. Training costs O(epochs * archive_size * dimension * hidden_width); inference
  costs O(dimension * hidden_width), with hidden_width at most eight.

Flow (pseudocode):
1. Validate the archive; compute logit means and scales.
2. Train the denoising network until the epoch cap or caller deadline.
3. Transform latent keys or route them through the uniform branch.
4. Decode the generated original-space keys with the caller's original decoder.

Important criteria:
- No objective predictions, domain features, pretrained weights or new dependencies.
- A protected branch preserves reachability, not finite-time optimization quality.
- This benchmark prototype adds no installed-package API.

Inputs/Outputs:
- Input: evaluated key vectors, latent random keys and an existing decoder.
- Output: decoded values through a bounded learned key distribution.
"""

from __future__ import annotations

import numpy as np
import time
from benchmarks.engines.mlp_surrogate import MlpSurrogate
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True, eq=False)
class KeyDistribution:
    """Fit and sample an elite-key distribution.

    Responsibilities:
    - Learn correlations between keys without reading problem features.
    - Produce finite keys in the original unit cube.

    Attributes:
        mean (np.ndarray): Per-coordinate logit means.
        scale (np.ndarray): Positive per-coordinate logit scales.
        weights (dict[str, np.ndarray] | None): Autoencoder parameters; None for the control.
        latent_mean (np.ndarray | None): Encoded elite latent means.
        latent_scale (np.ndarray | None): Encoded elite latent scales.
        epochs (int): Completed training steps.
        error (float): Final standardized reconstruction error.
    """

    mean: np.ndarray
    scale: np.ndarray
    weights: dict[str, np.ndarray] | None = None
    latent_mean: np.ndarray | None = None
    latent_scale: np.ndarray | None = None
    epochs: int = 0
    error: float = 0.0

    @staticmethod
    def coordinates(elite: np.ndarray) -> np.ndarray:
        """Validate the archive and return bounded logit coordinates.

        Args:
            elite (np.ndarray): At least four finite key rows with matching positive width.

        Returns:
            np.ndarray: Logit-transformed archive.

        Raises:
            ValueError: If the archive is malformed or holds values outside [0, 1).
        """
        rows = np.asarray(elite, dtype=np.float64)
        if rows.ndim != 2 or min(rows.shape) < 1 or len(rows) < 4:
            raise ValueError("'elite' must contain at least four equal-width key rows.")
        if not np.isfinite(rows).all() or np.any(rows < 0.0) or np.any(rows >= 1.0):
            raise ValueError("'elite' must contain finite keys in [0, 1).")
        bounded = np.clip(rows, 1e-6, 1.0 - 1e-6)
        return np.log(bounded) - np.log1p(-bounded)

    @classmethod
    def marginal(cls, elite: np.ndarray) -> KeyDistribution:
        """Construct the independent-coordinate non-neural control.

        Args:
            elite (np.ndarray): Good evaluated keys.

        Returns:
            KeyDistribution: Independent-coordinate distribution in logit space.

        Raises:
            ValueError: If the elite archive is invalid.
        """
        logits = cls.coordinates(elite)
        return cls(logits.mean(axis=0), np.maximum(logits.std(axis=0), 0.1))

    @classmethod
    def fit(
        cls, elite: np.ndarray, seed: int, epochs: int = 40, deadline: float | None = None
    ) -> KeyDistribution:
        """Fit a small denoising key generator within the caller's training allowance.

        Args:
            elite (np.ndarray): Good evaluated keys, one row per solution.
            seed (int): Training seed.
            epochs (int): Positive maximum training steps.
            deadline (float | None): Optional absolute monotonic deadline, checked every epoch.

        Returns:
            KeyDistribution: Fitted generator; the marginal control if no training step fits.

        Raises:
            ValueError: If the archive or epoch count is invalid.
        """
        if epochs < 1:
            raise ValueError("'epochs' must be positive.")
        model = cls.marginal(elite)
        centered = (cls.coordinates(elite) - model.mean) / model.scale
        model.train(centered, seed, epochs, deadline)
        return model

    def train(self, centered: np.ndarray, seed: int, epochs: int, deadline: float | None) -> None:
        """Optimize the reconstruction network with a cooperative per-epoch deadline.

        Args:
            centered (np.ndarray): Standardized archive rows.
            seed (int): Initialization and corruption seed.
            epochs (int): Positive training-step cap.
            deadline (float | None): Optional absolute monotonic deadline.
        """
        rng = np.random.default_rng(seed)
        width = min(8, centered.shape[1], len(centered) - 1)
        weights = {
            "w1": rng.normal(0.0, 1.0 / np.sqrt(centered.shape[1]), (centered.shape[1], width)),
            "b1": np.zeros(width),
            "w2": rng.normal(0.0, 0.1, (width, centered.shape[1])),
            "b2": np.zeros(centered.shape[1]),
        }
        state = (
            {name: np.zeros_like(value) for name, value in weights.items()},
            {name: np.zeros_like(value) for name, value in weights.items()},
        )
        for epoch in range(1, epochs + 1):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            noisy = centered + rng.normal(0.0, 0.1, centered.shape)
            grads = self.gradients(weights, noisy, centered)
            MlpSurrogate.adam(weights, grads, state, epoch)
            self.epochs = epoch
        if self.epochs:
            self.calibrate(weights, centered)

    @staticmethod
    def gradients(
        weights: dict[str, np.ndarray], noisy: np.ndarray, target: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Differentiate mean squared reconstruction loss and weight regularization.

        Args:
            weights (dict[str, np.ndarray]): Encoder and decoder parameters.
            noisy (np.ndarray): Corrupted standardized archive.
            target (np.ndarray): Clean standardized archive.

        Returns:
            dict[str, np.ndarray]: Gradients matching all parameter shapes.
        """
        hidden = np.tanh(noisy @ weights["w1"] + weights["b1"])
        residual = (hidden @ weights["w2"] + weights["b2"] - target) * (2.0 / target.size)
        back = (residual @ weights["w2"].T) * (1.0 - hidden**2)
        return {
            "w1": noisy.T @ back + 1e-3 * weights["w1"],
            "b1": back.sum(axis=0),
            "w2": hidden.T @ residual + 1e-3 * weights["w2"],
            "b2": residual.sum(axis=0),
        }

    def calibrate(self, weights: dict[str, np.ndarray], centered: np.ndarray) -> None:
        """Store fitted parameters and calibrate the latent sampling distribution.

        Args:
            weights (dict[str, np.ndarray]): Trained parameters.
            centered (np.ndarray): Standardized clean archive.
        """
        latent = np.tanh(centered @ weights["w1"] + weights["b1"])
        self.weights = weights
        self.latent_mean = latent.mean(axis=0)
        self.latent_scale = np.maximum(latent.std(axis=0), 0.05)
        self.error = float(np.mean((latent @ weights["w2"] + weights["b2"] - centered) ** 2))

    def transform(self, keys: np.ndarray) -> np.ndarray:
        """Map one latent vector into the original unit cube.

        Args:
            keys (np.ndarray): Latent random keys, one per original variable.

        Returns:
            np.ndarray: Generated keys in (0, 1).

        Raises:
            ValueError: If the input shape differs from the fitted dimension.
        """
        key = np.asarray(keys, dtype=np.float64)
        if key.shape != self.mean.shape:
            raise ValueError("Keys must match the fitted distribution's dimension.")
        normalized = np.sqrt(12.0) * (key - 0.5)
        if self.weights is not None:
            assert self.latent_mean is not None and self.latent_scale is not None
            width = len(self.latent_mean)
            latent = np.clip(self.latent_mean + normalized[:width] * self.latent_scale, -1.0, 1.0)
            normalized = latent @ self.weights["w2"] + self.weights["b2"]
        logits = np.clip(self.mean + self.scale * normalized, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-logits))


@dataclass(slots=True)
class NeuralKeyAdapter[Decoded]:
    """Compose a key generator with a caller-supplied decoder.

    Responsibilities:
    - Preserve a protected uniform route for the original key space.
    - Return the original decoder's output rather than a predicted cost.

    Attributes:
        decoder (Callable[..., Decoded]): Original problem decoder.
        distribution (KeyDistribution): Fitted key distribution.
        uniform (float): Probability of the protected uniform route.
    """

    decoder: Callable[..., Decoded]
    distribution: KeyDistribution
    uniform: float = 0.25

    def __post_init__(self) -> None:
        """Require a finite mixture probability retaining both routes.

        Raises:
            ValueError: If the uniform probability is outside (0, 1).
        """
        if not 0.0 < self.uniform < 1.0:
            raise ValueError("'uniform' must be finite and strictly between zero and one.")

    def protected(self, keys: np.ndarray) -> np.ndarray:
        """Encode an existing solution through the protected route.

        Args:
            keys (np.ndarray): Existing original-space keys.

        Returns:
            np.ndarray: Latent keys that reconstruct the supplied vector.
        """
        encoded = np.array(keys, dtype=np.float64)
        encoded[0] *= self.uniform
        return encoded

    def __call__(self, keys: np.ndarray) -> Decoded:
        """Return the original decoder's value for a generated or uniform key vector.

        Args:
            keys (np.ndarray): Latent random keys in [0, 1).

        Returns:
            Decoded: Original decoder's decoded value.
        """
        latent = np.array(keys, dtype=np.float64)
        if latent[0] < self.uniform:
            latent[0] /= self.uniform
            return self.decoder(latent)
        latent[0] = (latent[0] - self.uniform) / (1.0 - self.uniform)
        return self.decoder(self.distribution.transform(latent))
