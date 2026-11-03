"""Shared paired bootstrap confidence interval for development-screen comparisons.

Objective:
- Provide one exploratory paired-difference interval reused by F4 and F5 analysis.

Technical approach:
- Resample instance-level differences with replacement and report percentile bounds.

Flow (pseudocode):
1. Subtract control from candidate at matching instance identities.
2. Resample the differences with replacement ten thousand times.
3. Report the mean difference and its 95 percent bootstrap interval.

Important criteria:
- The caller is responsible for aligning candidate and control by instance identity
  before calling; this module only resamples an already-paired difference vector.

Inputs/Outputs:
- Input: paired per-instance candidate and control values.
- Output: mean difference and percentile interval bounds.
"""

from __future__ import annotations

import numpy as np

RESAMPLES = 10000


def paired_interval(differences: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Bootstrap a paired-difference vector's mean and 95 percent interval.

    Args:
        differences (np.ndarray): Nonempty per-instance candidate-minus-control values.
        seed (int): Bootstrap resampling seed.

    Returns:
        dict[str, float]: Mean difference, `ci_low` and `ci_high`.

    Raises:
        ValueError: If `differences` is empty.
    """
    if len(differences) == 0:
        raise ValueError("Paired bootstrap requires a nonempty difference vector.")
    picks = np.random.default_rng(seed).integers(0, len(differences), (RESAMPLES, len(differences)))
    means = differences[picks].mean(axis=1)
    return {
        "difference": float(differences.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }
