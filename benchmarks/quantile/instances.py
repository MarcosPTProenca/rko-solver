"""Versioned deterministic synthetic tasks for contextual quantile experiments.

Objective:
- Generate distinct training, validation and test tasks with explicit content identities.

Technical approach:
- Use caller-owned local numerical generators and condition covering rows on nonempty support.

Flow (pseudocode):
1. Validate the family, size and generation seed.
2. Generate unit-square coordinates or binary covering incidence with positive costs.
3. Validate through exact task adapters and fingerprint original data.

Important criteria:
- Public benchmark data are not substituted by these synthetic distributions.

Inputs/Outputs:
- Input: declared task family, size and seed.
- Output: validated immutable tasks and stable content fingerprints.
"""

import hashlib
import json
import numpy as np
from benchmarks.quantile.tasks import Cover
from benchmarks.quantile.tasks import Tour


class Instances:
    """Generate the two declared synthetic task distributions without global randomness.

    Responsibilities:
    - Keep generation semantics versioned and local to an explicit seed.
    - Identify original task content independently of labels and measured outcomes.
    """

    VERSION = "quantile-instances-v1"

    @staticmethod
    def make(family: str, size: int, seed: int) -> Tour | Cover:
        """Generate one immutable task from the declared distribution.

        Args:
            family (str): 'tour' or 'cover'.
            size (int): City or row count, between two and five hundred.
            seed (int): Nonnegative local generator initialization.

        Returns:
            Tour | Cover: Checked task with deterministic original data.

        Raises:
            ValueError: If family, size or seed is unsupported.
        """
        if family not in ("tour", "cover"):
            raise ValueError("Instance family must be 'tour' or 'cover'.")
        if type(size) is not int or not 2 <= size <= 500:
            raise ValueError("Instance size must be an integer between 2 and 500.")
        if type(seed) is not int or seed < 0:
            raise ValueError("Instance seed must be a nonnegative integer.")
        rng = np.random.Generator(np.random.PCG64(seed))
        if family == "tour":
            return Tour(rng.random((size, 2)))
        return Instances._make_cover(rng, size)

    @staticmethod
    def _make_cover(rng: np.random.Generator, size: int) -> Cover:
        """Generate a sparse, fully coverable incidence matrix with positive column costs.

        Args:
            rng (np.random.Generator): Owned local covering-instance generator.
            size (int): Row count; the matrix has twice as many columns.

        Returns:
            Cover: Checked covering task.
        """
        matrix = rng.random((size, 2 * size)) < 0.05
        for row in matrix:
            while not row.any():
                row[:] = rng.random(2 * size) < 0.05
        return Cover(matrix, rng.integers(1, 101, 2 * size))

    @classmethod
    def fingerprint(cls, task: Tour | Cover) -> str:
        """Hash versioned original task data without using measured reference objectives.

        Args:
            task (Tour | Cover): Validated task to identify.

        Returns:
            str: Stable hexadecimal digest of declared family, shapes and original values.
        """
        family = "tour" if isinstance(task, Tour) else "cover"
        arrays = (task.coordinates,) if isinstance(task, Tour) else (task.incidence, task.costs)
        digest = hashlib.sha256(json.dumps((cls.VERSION, family)).encode())
        for values in arrays:
            digest.update(json.dumps((values.shape, str(values.dtype))).encode())
            digest.update(values.tobytes())
        return digest.hexdigest()
