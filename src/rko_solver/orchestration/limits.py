"""Upper bounds on durations, evaluation counts and allocation sizes.

Objective:
- Reject settings that no machine can satisfy, or that would exhaust memory or panic a native
  engine, before any engine starts, with the same message on every engine.

Technical approach:
- Four named bounds: seconds (one year, which also keeps float seconds convertible to a
  native `Duration`), evaluations (the largest signed 64-bit integer), population-like sizes
  (one million) and population-times-dimension cells (one hundred million `float64` values,
  about 800 MB per population).
- Checks are pure functions of the supplied numbers, so offline values and every reachable
  online grid value can be validated without building an algorithm.

Flow (pseudocode):
1. `RKO.solve` validates `time_limit`, `auto_time` and `max_evaluations`.
2. The parameter validator checks each population-like value and its cells against the
   problem dimension, for offline values and for the largest value of each online axis.
3. The solver checks the shared pool the same way.

Important criteria:
- Bounds are generous: they exist to stop absurd or hostile values, not to restrict real runs.
- Boolean values are never accepted as numbers.

Inputs/Outputs:
- Input: a named number and, for allocations, the key dimension.
- Output: no return value, or a `ValueError` naming the setting and the bound.
"""

import math
from numbers import Real


class RunLimits:
    """Validate durations, counts and allocation sizes against fixed upper bounds.

    Responsibilities:
    - Hold the bounds in one place shared by every engine.
    - Raise a setting-specific `ValueError` for values above them.

    Attributes:
        MAX_SECONDS (float): Largest accepted duration in seconds (one year).
        MAX_EVALUATIONS (int): Largest accepted evaluation limit.
        MAX_POPULATION (int): Largest accepted population, pool or swarm size.
        MAX_CELLS (int): Largest accepted `population * dimension` product.
    """

    MAX_SECONDS = 31_536_000.0
    MAX_EVALUATIONS = 2**63 - 1
    MAX_POPULATION = 1_000_000
    MAX_CELLS = 100_000_000
    POPULATION_KEYS = frozenset({"p", "size_pop", "p_size"})

    @classmethod
    def seconds(cls, name: str, value: object) -> None:
        """Require a positive, finite duration no longer than `MAX_SECONDS`.

        Args:
            name (str): Setting name used in the message.
            value (object): Candidate duration.

        Raises:
            ValueError: If the value is not a real number in `(0, MAX_SECONDS]`.
        """
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not 0 < float(value) <= cls.MAX_SECONDS
        ):
            raise ValueError(
                f"'{name}' must be positive, finite and at most {cls.MAX_SECONDS:g} seconds."
            )

    @classmethod
    def evaluations(cls, value: object) -> None:
        """Require `None` or a positive integer evaluation limit within 64 bits.

        Args:
            value (object): Candidate `max_evaluations`, where `None` means no limit.

        Raises:
            ValueError: If the value is neither `None` nor an integer in
                `[1, MAX_EVALUATIONS]`.
        """
        if value is not None and (type(value) is not int or not 1 <= value <= cls.MAX_EVALUATIONS):
            raise ValueError(
                f"'max_evaluations' must be an integer from 1 to {cls.MAX_EVALUATIONS}, or None."
            )

    @classmethod
    def allocation(cls, name: str, size: float, dimension: int) -> None:
        """Bound a population-like size and the memory its key vectors need.

        Args:
            name (str): Setting name used in the message.
            size (float): Number of individuals, entries or particles.
            dimension (int): Number of random keys per individual.

        Raises:
            ValueError: If `size` exceeds `MAX_POPULATION` or `size * dimension` exceeds
                `MAX_CELLS`.
        """
        if size > cls.MAX_POPULATION:
            raise ValueError(f"'{name}' must be at most {cls.MAX_POPULATION}.")
        if size * dimension > cls.MAX_CELLS:
            raise ValueError(
                f"'{name}' times the key dimension ({dimension}) must be at most {cls.MAX_CELLS}."
            )
