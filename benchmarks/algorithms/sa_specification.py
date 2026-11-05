"""Export production offline simulated annealing specifications without solving.

Objective:
- Prevent repository-only native studies from silently using hardcoded benchmark parameters.

Technical approach:
- Snapshot parameters through the production native-runner specification loader.
- Encode the ordered vector for the private benchmark and fingerprint its values.

Flow (pseudocode):
1. Construct offline simulated annealing configuration for the requested dimension.
2. Snapshot the same ordered parameters used by the packaged engine.
3. Emit native arguments and verify the executable's parameter echo.

Important criteria:
- No objective is evaluated and no public option or packaged default changes.
- Legacy screen cohorts and their binaries remain untouched.

Inputs/Outputs:
- Input: positive random-key dimension.
- Output: immutable ordered parameter vector, fingerprint and benchmark arguments.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from rko_solver.engines.rust.runner import NativeRunner
from rko_solver.orchestration.types import RunConfig


@dataclass(frozen=True, slots=True)
class SaSpecification:
    """Represent the actual packaged offline simulated annealing parameters.

    Responsibilities:
    - Preserve native parameter ordering and round-trip numerical values.
    - Refuse a native executable that reports a different specification.

    Attributes:
        values (tuple[float, ...]): Ordered native offline parameter vector.
    """

    values: tuple[float, ...]

    @classmethod
    def for_dimension(cls, n_variables: int) -> SaSpecification:
        """Snapshot the production offline parameter vector without executing a search.

        Args:
            n_variables (int): Positive problem dimension.

        Returns:
            SaSpecification: Immutable production parameter snapshot.

        Raises:
            ValueError: If the dimension or packaged specification is invalid.
            OSError: If packaged parameters cannot be read.
        """
        if n_variables < 1:
            raise ValueError("specification dimension must be positive")
        config = RunConfig(control=0, pool_size=10, active_metaheuristics=("SA",))
        specifications, _ = NativeRunner(config, n_variables, None, None).specs()
        return cls(tuple(specifications[0][1]))

    @property
    def sha256(self) -> str:
        """Return a deterministic fingerprint of the ordered numerical values.

        Returns:
            str: Content digest for the canonical parameter vector.
        """
        encoded = json.dumps(self.values, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    def native_arguments(self) -> list[str]:
        """Encode round-trippable values for the repository-only native benchmark.

        Returns:
            list[str]: Parameter option and its ordered comma-separated values.
        """
        return ["--sa-parameters", ",".join(repr(value) for value in self.values)]

    def verify(self, native_values: Sequence[float]) -> None:
        """Refuse native output that does not reproduce the exported specification.

        Args:
            native_values (Sequence[float]): Parameters reported by the native executable.

        Raises:
            ValueError: If ordering or numerical values differ.
        """
        if tuple(native_values) != self.values:
            raise ValueError("native specification differs from the packaged offline specification")
