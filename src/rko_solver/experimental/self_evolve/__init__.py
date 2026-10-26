"""Opt-in decoder evolution contracts; generated Python executes only in qualified workers.

Objective:
- Expose research artifact and problem contracts without altering the RKO engine.
Technical approach:
- Keep imports side-effect free and source loading non-executing.
Flow (pseudocode):
1. Validate an existing seed and independent evaluation contract.
2. Use the explicit isolated runner for execution and offline replay.
Important criteria:
- This namespace is experimental, not a production deployment API.
Inputs/Outputs:
- Input: explicit artifacts and problem instances.
- Output: validated descriptors and independently recomputable solution evidence.
"""

from .artifacts import DecoderArtifact
from .problems import Problem

__all__ = ["DecoderArtifact", "Problem"]
