"""Optional runtime provenance for model validation.

Objective:
- Preserve reproducible complete benchmark evidence.

Technical approach:
- Separate observational bookkeeping from timed solver execution.

Flow (pseudocode):
1. Read original runtime or campaign data.
2. Validate all planned evidence before aggregation.
3. Return explicit reproducible provenance or complete summaries.

Important criteria:
- Never omit errors, missing runs or unsupported source identity.

Inputs/Outputs:
- Input: original runtime or campaign observations.
- Output: portable complete benchmark evidence.
"""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path
from typing import Any


class CampaignProvenance:
    """Capture reproducible optional-runtime identity.

    Responsibilities:
    - Preserve CPU fallback and missing-package version semantics.
    """

    @staticmethod
    def runtime() -> dict[str, Any]:
        """Capture optional package versions and portable CPU identity.

        Returns:
            dict[str, Any]: Validated original campaign data.
        """
        try:
            cpu_model = next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            )
        except OSError, StopIteration:
            cpu_model = platform.processor() or "unavailable"
        versions = {}
        for package in ("rko-solver", "numpy", "scipy", "pyomo", "pulp", "ortools"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        return {"cpu_model": cpu_model, "versions": versions}
