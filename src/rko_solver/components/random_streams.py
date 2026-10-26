"""Context-local random streams for library-owned search operations.

Objective:
- Isolate independent searches without reseeding caller-global generators.

Technical approach:
- Own Python Random and NumPy RandomState instances per search.
- Resolve streams through ContextVar, with global fallback for standalone operators.

Flow (pseudocode):
1. Construct independent generators from the assigned algorithm seed.
2. Activate the pair while constructing or executing that search.
3. Restore the enclosing context even if construction or execution raises.

Important criteria:
- RandomState intentionally preserves legacy seeded numeric trajectories.
- User callbacks using global RNGs must manage their own random state.
- Resolve Python generators once per numeric kernel, not once per gene.

Inputs/Outputs:
- Input: an assigned seed and a scoped construction/search operation.
- Output: isolated generator draws without changing global RNG state.
"""

from __future__ import annotations

import numpy as np
import random
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType


class RandomStreams:
    """Own and activate the two legacy-compatible generators for one search.

    Responsibilities:
    - Preserve seeded draw ordering without sharing mutable generator state.
    - Restore nested contexts across normal return and exceptional exits.

    Attributes:
        seed (int | None): Assigned seed for result provenance.
        python (random.Random): Private Python generator.
        numpy (np.random.RandomState): Private NumPy legacy generator.
    """

    _current: ContextVar[RandomStreams | None] = ContextVar("rko_random_streams", default=None)

    def __init__(self, seed: int | None) -> None:
        """Initialize independently owned generators.

        Args:
            seed (int | None): Assigned uint32 seed, or system entropy.

        Raises:
            ValueError: If NumPy rejects the supplied seed.
        """
        self.seed = seed
        self.python = random.Random(seed)
        self.numpy = np.random.RandomState(seed)

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Activate this search's streams without modifying global generators.

        Yields:
            None: Execution within the current search's generator context.
        """
        token = self._current.set(self)
        try:
            yield
        finally:
            self._current.reset(token)

    @classmethod
    def python_rng(cls) -> random.Random | ModuleType:
        """Resolve the current Python stream or preserve standalone behavior.

        Returns:
            random.Random | ModuleType: Private stream or Python's global module.
        """
        current = cls._current.get()
        return random if current is None else current.python

    @classmethod
    def numpy_rng(cls) -> np.random.RandomState | ModuleType:
        """Resolve the current NumPy stream or preserve standalone behavior.

        Returns:
            np.random.RandomState | ModuleType: Private stream or global NumPy RNG.
        """
        current = cls._current.get()
        return np.random if current is None else current.numpy
