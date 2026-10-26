"""Preplanned paired callback boundary protection.

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

import gc
import numpy as np
import random
from numbers import Real
from rko_solver.orchestration.search import SearchSession
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from .modeling_overhead import BoundaryWorkload
    from .modeling_overhead import ModelOverheadProbe


class ModelBoundaryProbe:
    """Measure paired identical-work callback boundary controls.

    Responsibilities:
    - Keep measured work and clock boundaries unchanged.
    - Preserve reproducible ordering, controls and failure reporting.
    """

    def __init__(
        self,
        probe: type[ModelOverheadProbe],
        baseline: type[SearchSession],
        repeats: int,
        target_seconds: float,
    ) -> None:
        """Own the original paired order and calibration policy.

        Args:
            probe (type[ModelOverheadProbe]): Original shared measurement operations.
            baseline (type[SearchSession]): Exact original search session class.
            repeats (int): Preplanned paired trials.
            target_seconds (float): Fixed-work calibration target.

        """
        self.probe = probe
        self.baseline = baseline
        self.repeats = repeats
        self.target_seconds = target_seconds
        self.generator = random.Random(20261025)

    @staticmethod
    def settings(repeats: int, target_seconds: float) -> None:
        """Validate original paired-work settings.

        Args:
            repeats (int): Preplanned paired repetitions.
            target_seconds (float): Positive finite calibration target.


        Raises:
            ValueError: If repetitions or calibration settings are invalid.
        """
        if (
            type(repeats) is not int
            or repeats < 3
            or isinstance(target_seconds, bool)
            or (not isinstance(target_seconds, Real))
            or (not np.isfinite(target_seconds))
            or (target_seconds <= 0)
        ):
            raise ValueError(
                "At least three pairs and a positive finite calibration target are required."
            )

    def _count(
        self, control: type[SearchSession], workload: BoundaryWorkload, archive: bool
    ) -> int:
        """Calibrate unchanged fixed work without altering warmups or iteration caps.

        Args:
            control (type[SearchSession]): Exact control boundary implementation.
            workload (BoundaryWorkload): Unchanged original callback workload.
            archive (bool): Original control archive policy.

        Returns:
            int: Original measured observations or validated settings.
        """
        calibration_count = 64
        calibration = self.probe.sample(control, workload, calibration_count, archive=archive)
        while calibration < 0.02 and calibration_count < 65536:
            calibration_count *= 4
            calibration = self.probe.sample(control, workload, calibration_count, archive=archive)
        count = max(64, min(1000000, int(calibration_count * self.target_seconds / calibration)))
        return count

    def measure(self, workload: BoundaryWorkload) -> dict[str, Any]:
        """Measure original paired samples in their preplanned randomized order.

        Args:
            workload (BoundaryWorkload): Unchanged original callback workload.

        Returns:
            dict[str, Any]: Original measured observations or validated settings.

        Raises:
            AssertionError: If an original workload result changes.
        """
        control = SearchSession if workload.generated else self.baseline
        archive = not workload.generated
        self.probe.sample(control, workload, 64, archive=archive)
        self.probe.sample(SearchSession, workload, 64)
        count = self._count(control, workload, archive)
        self.probe.sample(control, workload, count, archive=archive)
        self.probe.sample(SearchSession, workload, count)
        pairs = []
        for _ in range(self.repeats):
            order = [0, 1]
            self.generator.shuffle(order)
            timings = {}
            for index in order:
                gc.collect()
                timings[index] = self.probe.sample(
                    control if index == 0 else SearchSession,
                    workload,
                    count,
                    archive=archive if index == 0 else True,
                )
            pairs.append({"control_seconds": timings[0], "candidate_seconds": timings[1]})
        return {
            "workload": workload.name,
            "dimension": len(workload.keys),
            "kind": "enabled_archive" if workload.generated else "disabled_vs_original",
            "evaluations_per_sample": count,
            "expected_cost": workload.expected,
            "pairs": pairs,
            "summary": self.probe.paired_summary(pairs, protection=not workload.generated),
        }
