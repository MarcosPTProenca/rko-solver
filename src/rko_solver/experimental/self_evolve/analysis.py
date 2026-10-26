"""Paired instance-level estimates for frozen decoder artifact comparisons.

Objective:
- Avoid treating repeated solver seeds or adaptively selected programs as independent evidence.
Technical approach:
- Average seeds within each instance, bootstrap paired instance differences and test sign flips.
- Apply Holm correction to the entire predeclared primary comparison family.
Flow (pseudocode):
1. Validate complete finite paired loss matrices in the fixed [0, 1] study utility.
2. Resample instances for a percentile interval; enumerate or sample paired sign flips.
3. Return effect/uncertainty and explicit assumptions, never an automatic deployment decision.
Important criteria:
- Sign flips require null exchangeability/symmetry; bootstrap requires independent instances.
- Bootstrap intervals/bounds are approximate and can under-cover in small/degenerate panels.
Inputs/Outputs:
- Input: instance-by-seed loss matrices from frozen, non-adaptively inspected test data.
- Output: descriptive paired effect, marginal interval, one-sided p-value and corrected p-values.
"""

from __future__ import annotations

import math
import numpy as np
from ...components.random_streams import RandomStreams
from typing import Any


class PairedAnalysis:
    """Estimate artifact-level paired effects without seed pseudoreplication.

    Responsibilities:
    - Validate panels and make stochastic numerical analysis reproducible and memory-bounded.

    Attributes:
        bootstrap_samples (int): Number of instance-cluster bootstrap draws.
        permutation_samples (int): Monte Carlo draws when exact sign enumeration is too large.
        random_state (int): Private analysis seed, independent of solver/proposer seeds.
        family_size (int): Predeclared multiplicity for the one-sided bootstrap lower bound.
    """

    def __init__(
        self,
        *,
        bootstrap_samples: int = 10000,
        permutation_samples: int = 9999,
        random_state: int = 42,
        family_size: int = 1,
    ) -> None:
        """Freeze bounded numerical-analysis settings.

        Args:
            bootstrap_samples (int): Between 100 and 50000 resamples.
            permutation_samples (int): Between 100 and 50000 resamples.
            random_state (int): Explicit uint32 seed for owned streams.
            family_size (int): Number of predeclared primary comparisons for the lower bound.

        Raises:
            ValueError: If settings violate the analysis work envelope.
        """
        if (
            type(bootstrap_samples) is not int
            or not 100 <= bootstrap_samples <= 50000
            or type(permutation_samples) is not int
            or not 100 <= permutation_samples <= 50000
            or type(random_state) is not int
            or not 0 <= random_state < 2**32
            or type(family_size) is not int
            or not 1 <= family_size <= 100
        ):
            raise ValueError("Invalid analysis resampling settings or seed.")
        self.bootstrap_samples = bootstrap_samples
        self.permutation_samples = permutation_samples
        self.random_state = random_state
        self.family_size = family_size

    @staticmethod
    def strata_indices(n: int, strata: list[str] | None = None) -> dict[str, list[int]]:
        """Validate predeclared strata with enough independent units for resampling.

        Args:
            n (int): Number of independent instances.
            strata (list[str] | None): Fixed labels; None declares one mixture stratum.

        Returns:
            dict[str, list[int]]: Instance positions grouped by stratum.

        Raises:
            ValueError: If labels mismatch or a stratum contains fewer than two instances.
        """
        labels = ["all"] * n if strata is None else strata
        if len(labels) != n or any(not isinstance(s, str) or not 0 < len(s) <= 64 for s in labels):
            raise ValueError("Strata must supply one bounded nonempty label per instance.")
        groups: dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            groups.setdefault(label, []).append(index)
        if any(len(indices) < 2 for indices in groups.values()):
            raise ValueError("Each bootstrap stratum requires at least two independent instances.")
        return groups

    def compare(
        self,
        baseline: list[list[float]],
        candidate: list[list[float]],
        *,
        strata: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute an instance-paired positive-is-better loss difference and uncertainty.

        Args:
            baseline (list[list[float]]): Complete instance-by-seed losses in [0, 1].
            candidate (list[list[float]]): Same instances/seeds in the same frozen order.
            strata (list[str] | None): Frozen sampling strata, resampled without changing weights.

        Returns:
            dict[str, Any]: Paired effect, marginal percentile interval and one-sided p-value.

        Raises:
            ValueError: If observations are missing, nonfinite, out of range or unpaired.
        """
        try:
            left, right = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
        except (ValueError, TypeError) as error:
            raise ValueError("Loss panels must be rectangular numeric matrices.") from error
        self._validate_panels(left, right)
        differences = (left - right).mean(axis=1)
        n = len(differences)
        groups = self.strata_indices(n, strata)
        effect = float(differences.mean())
        bootstrap = self._bootstrap(differences, groups)
        interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        p_value, exact, draws = self._sign_flip(differences, effect)
        return {
            "n_instances": n,
            "strata_counts": {label: len(indices) for label, indices in groups.items()},
            "seeds_per_instance": left.shape[1],
            "baseline_mean_loss": float(left.mean()),
            "candidate_mean_loss": float(right.mean()),
            "instance_effects": differences.tolist(),
            "effect_mean": effect,
            "confidence_interval_95": interval,
            "interval_method": "paired_instance_percentile",
            "interval_scope": "marginal_not_simultaneous",
            "p_value": p_value,
            "familywise_lower_bound_95": float(np.quantile(bootstrap, 0.05 / self.family_size)),
            "lower_bound_method": "Bonferroni_one_sided_percentile_approximation",
            "primary_family_size": self.family_size,
            "alternative": "baseline_loss_minus_candidate_loss_greater_than_zero",
            "permutation_method": "exact_paired_sign_flip"
            if exact
            else "randomized_paired_sign_flip",
            "permutation_draws": draws,
            "assumptions": "Independent instances and exchangeable/symmetric paired null effects. "
            "Seeds are averaged within instances, not counted as independent units.",
        }

    @staticmethod
    def _validate_panels(left: np.ndarray, right: np.ndarray) -> None:
        """Validate paired matrix dimensions before checking the numeric domains.

        Args:
            left (np.ndarray): Baseline matrix.
            right (np.ndarray): Candidate matrix.

        Raises:
            ValueError: If matrices are not bounded paired loss panels.
        """
        if (
            left.ndim != 2
            or left.shape != right.shape
            or not 2 <= left.shape[0] <= 1000
            or not 1 <= left.shape[1] <= 100
            or not PairedAnalysis._bounded_losses(left)
            or not PairedAnalysis._bounded_losses(right)
        ):
            raise ValueError("Need complete finite paired instance-by-seed losses in [0, 1].")

    @staticmethod
    def _bounded_losses(panel: np.ndarray) -> bool:
        """Check finite entries against the fixed utility domain.

        Args:
            panel (np.ndarray): Numeric loss panel.

        Returns:
            bool: Whether all losses are finite and in [0, 1].
        """
        return bool(np.isfinite(panel).all() and not np.any(panel < 0) and not np.any(panel > 1))

    def _bootstrap(self, differences: np.ndarray, groups: dict[str, list[int]]) -> np.ndarray:
        """Resample paired instance effects in fixed-size stratified batches.

        Args:
            differences (np.ndarray): Seed-averaged paired effects.
            groups (dict[str, list[int]]): Ordered stratum positions.

        Returns:
            np.ndarray: Bootstrap means in original draw order.
        """
        n = len(differences)
        rng = RandomStreams(self.random_state).numpy
        bootstrap = np.empty(self.bootstrap_samples)
        for start in range(0, self.bootstrap_samples, 256):
            stop = min(start + 256, self.bootstrap_samples)
            indices = np.empty((stop - start, n), dtype=int)
            for positions in groups.values():
                draws = rng.randint(0, len(positions), size=(stop - start, len(positions)))
                indices[:, positions] = np.asarray(positions)[draws]
            bootstrap[start:stop] = differences[indices].mean(axis=1)
        return bootstrap

    def _sign_flip(self, differences: np.ndarray, effect: float) -> tuple[float, bool, int]:
        """Enumerate or sample paired signs using an independent fixed stream.

        Args:
            differences (np.ndarray): Seed-averaged paired effects.
            effect (float): Observed paired mean.

        Returns:
            tuple[float, bool, int]: One-sided p-value, exactness and draw count.
        """
        n = len(differences)
        tolerance = 100 * np.finfo(float).eps * max(1.0, abs(effect))
        count = 0
        exact = n <= 16
        draws = 1 << n if exact else self.permutation_samples
        rng = RandomStreams(self.random_state ^ 0x51F15E).numpy
        for start in range(0, draws, 256):
            stop = min(start + 256, draws)
            if exact:
                masks = np.arange(start, stop, dtype=np.uint64)[:, None]
                bits = np.arange(n, dtype=np.uint64)[None, :]
                signs = 2 * ((masks >> bits) & 1).astype(float) - 1
            else:
                signs = 2 * rng.randint(0, 2, size=(stop - start, n)) - 1
            simulated = (signs * differences).mean(axis=1)
            count += int(np.count_nonzero(simulated >= effect - tolerance))
        p_value = count / draws if exact else (count + 1) / (draws + 1)
        return p_value, exact, draws

    @staticmethod
    def holm(p_values: list[float]) -> list[float]:
        """Adjust all predeclared primary p-values with Holm's step-down procedure.

        Args:
            p_values (list[float]): Entire primary hypothesis family, not selected significant hits.

        Returns:
            list[float]: Adjusted p-values in the original comparison order.

        Raises:
            ValueError: If the family is empty or contains invalid probabilities.
        """
        if not p_values or any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in p_values
        ):
            raise ValueError("Holm correction requires finite p-values in [0, 1].")
        order = sorted(range(len(p_values)), key=p_values.__getitem__)
        adjusted = [0.0] * len(p_values)
        maximum = 0.0
        for rank, index in enumerate(order):
            maximum = max(maximum, min(1.0, (len(order) - rank) * p_values[index]))
            adjusted[index] = maximum
        return adjusted
