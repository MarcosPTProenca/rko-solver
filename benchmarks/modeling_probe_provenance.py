"""Exact overhead probe source and callback provenance.

Objective:
- Preserve reproducible component measurements and explicit controls.

Technical approach:
- Separate clock-bound measured work from setup and source bookkeeping.

Flow (pseudocode):
1. Prepare the unchanged original inputs and control settings.
2. Execute the original fixed work or capture exact source identity.
3. Return complete observations without claiming an unmeasured speedup.

Important criteria:
- Preserve paired order, warmups, fixed work, cleanup and failure propagation.

Inputs/Outputs:
- Input: original workload and preplanned measurement policy.
- Output: complete source-identified component observations.
"""

from __future__ import annotations

import hashlib
import numpy as np
import os
import platform
import subprocess
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    pass


class ModelProbeProvenance:
    """Own exact candidate source hashes and unchanged handwritten callback controls.

    Responsibilities:
    - Keep measured work and clock boundaries unchanged.
    - Preserve reproducible ordering, controls and failure reporting.
    """

    @staticmethod
    def create(
        revision: str, repeats: int, target_seconds: float, baseline_hash: str, threshold: float
    ) -> dict[str, Any]:
        """Capture exact source identity before measuring any work.

        Args:
            revision (str): Exact original source revision.
            repeats (int): Preplanned paired trials.
            target_seconds (float): Fixed-work calibration target.
            baseline_hash (str): Original search boundary source digest.
            threshold (float): Declared disabled-path protection ratio.

        Returns:
            dict[str, Any]: Original measured observations or validated settings.

        Raises:
            OSError: If sources cannot be read.
            RuntimeError: If handwritten callback source differs from the original control.
        """
        root = Path(__file__).resolve().parents[1]
        source_paths = [
            Path(__file__),
            *(root / "src/rko_solver").rglob("*.py"),
            *(root / "src/rko_solver/parameters").glob("*.toml"),
            root / "src/rko_solver/examples/instances/kp50.txt",
            root / "pyproject.toml",
            root / "uv.lock",
            root / ".python-version",
            root / "benchmarks/modeling_cases.py",
            root / "benchmarks/modeling_schema.py",
            *(root / "benchmarks").glob("modeling*.py"),
        ]
        hashes = {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
            if "tests" not in path.relative_to(root).parts
        }
        callback_sources = {}
        for relative in (
            "src/rko_solver/examples/instance_pipeline.py",
            "src/rko_solver/examples/run_kp_cpp_instances.py",
        ):
            original = subprocess.check_output(["git", "show", f"{revision}:{relative}"])
            digest = hashlib.sha256(original).hexdigest()
            if digest != hashes[relative]:
                raise RuntimeError(
                    "Handwritten callback source changed; paired control is invalid."
                )
            callback_sources[relative] = digest
        data = {
            "schema_version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "baseline_revision": subprocess.check_output(
                ["git", "rev-parse", revision], text=True, encoding="utf-8"
            ).strip(),
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
            "handwritten_callback_sources_unchanged": callback_sources,
            "baseline_search_sha256": baseline_hash,
            "source_sha256": hashes,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "available_cpus": os.process_cpu_count(),
            "repeats": repeats,
            "target_seconds": target_seconds,
            "disabled_protection_ratio": threshold,
            "gc_disabled_during_samples": True,
            "calibration_minimum_seconds": 0.02,
            "warmup_evaluations": 64,
            "boundary": [],
        }
        return data
