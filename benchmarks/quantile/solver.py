"""Explicit bridge from checked quantile constructions to existing solver callbacks.

Objective:
- Reuse the public callback and initial-key interfaces without changing solver defaults.

Technical approach:
- Carry immutable checked candidates between decoding and objective evaluation.
- Normalize only an exact unit endpoint produced by a solver operator.

Flow (pseudocode):
1. Copy solver keys and record exact upper-endpoint normalization.
2. Decode through the strict task mapping.
3. Recheck the original candidate payload independently when scoring.

Important criteria:
- No mutable last-candidate state or implicit normalization of arbitrary invalid inputs.

Inputs/Outputs:
- Input: solver keys and immutable candidate payloads.
- Output: checked candidate objects and finite minimization costs.
"""

import numpy as np
from benchmarks.quantile.codec import Candidate
from benchmarks.quantile.codec import TrajectoryCodec
from dataclasses import dataclass
from dataclasses import replace


@dataclass(frozen=True, slots=True)
class SolverBridge:
    """Expose strict quantile construction through ordinary solver callbacks.

    Responsibilities:
    - Record exact upper-endpoint normalization on each immutable candidate.
    - Re-evaluate the original decoded payload rather than a mutable last result.

    Attributes:
        codec (TrajectoryCodec): Fixed task and numerical construction configuration.
    """

    codec: TrajectoryCodec

    def decode(self, keys: np.ndarray) -> Candidate:
        """Normalize only exact upper endpoints and decode an owned key snapshot.

        Args:
            keys (np.ndarray): Candidate keys emitted by a solver operator.

        Returns:
            Candidate: Independently checked payload with endpoint-repair count.

        Raises:
            ValueError: If keys or the construction fail the strict numerical contract.
        """
        values = np.array(keys, dtype=np.float64, copy=True)
        endpoints = values == 1.0
        values[endpoints] = np.nextafter(1.0, 0.0)
        candidate = self.codec.decode(values)
        return replace(candidate, boundary_repairs=int(np.count_nonzero(endpoints)))

    def cost(self, candidate: Candidate) -> float:
        """Independently check the original payload's trace, solution and objective.

        Args:
            candidate (Candidate): Immutable payload from the matching decoder call.

        Returns:
            float: Exact finite minimization cost for this payload.

        Raises:
            ValueError: If payload identity, feasibility or its recorded cost is inconsistent.
        """
        if self.codec.task.finish(candidate.trace) != candidate.solution:
            raise ValueError("Candidate trace and solution do not match.")
        checked = self.codec.task.check(candidate.solution)
        if checked != candidate.cost:
            raise ValueError("Candidate cost does not match independent evaluation.")
        return checked
