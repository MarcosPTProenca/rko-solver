"""Explicit continuous-restoration feature cost.

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

import numpy as np
import statistics
import time
from .modeling_cases import PracticalCase
from .modeling_cases import PracticalModels
from rko_solver.modeling.linear import CompiledLinearModel
from rko_solver.modeling.linear import LinearRow
from rko_solver.modeling.linear import LinearVariable
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    pass


class ModelRestorationProbe:
    """Measure changed-assignment restoration cost independently of engine speed claims.

    Responsibilities:
    - Keep measured work and clock boundaries unchanged.
    - Preserve reproducible ordering, controls and failure reporting.
    """

    @staticmethod
    def _snapshots(case: PracticalCase) -> list[CompiledLinearModel]:
        """Build the original box and continuous-policy snapshots.

        Args:
            case (PracticalCase): Unchanged original mathematical fixture.

        Returns:
            list[CompiledLinearModel]: Original measured observations or validated settings.

        Raises:
            ValueError: If the original fixture is numerically unsupported.
        """
        variants = [
            CompiledLinearModel(
                tuple(
                    LinearVariable(v.name, v.name, v.lower, v.upper, v.kind) for v in case.variables
                ),
                case.objective,
                tuple(
                    LinearRow(row.name, row.coefficients, row.lower, row.upper) for row in case.rows
                ),
                maximize=case.maximize,
                repair=policy,
            )
            for policy in ("none", "continuous")
        ]
        return variants

    @staticmethod
    def run() -> list[dict[str, Any]]:
        """Measure original fixed-key restoration work and independently checked feasibility.

        Returns:
            list[dict[str, Any]]: Original measured observations or validated settings.

        Raises:
            AssertionError: If objective or feasibility checks disagree.
            OSError: If bundled fixture input cannot be read.
        """
        records = []
        for case in PracticalModels.cases():
            if case.name not in ("production_lp", "production_setup", "transport_equalities"):
                continue
            variants = ModelRestorationProbe._snapshots(case)
            keys = np.random.default_rng(2026).random((64, variants[0].n_keys))
            times = [[], []]
            sweeps = [0, 0]
            feasible = [0, 0]
            for index, chromosome in enumerate(keys):
                for variant in (0, 1) if index % 2 == 0 else (1, 0):
                    began = time.perf_counter()
                    result = variants[variant].decode(chromosome)
                    times[variant].append(time.perf_counter() - began)
                    checked = PracticalModels.check(
                        case,
                        {
                            v.name: value
                            for v, value in zip(case.variables, result.values, strict=True)
                        },
                    )
                    assert result.feasible == checked["feasible"]
                    assert abs(result.objective - checked["objective"]) <= 1e-08
                    feasible[variant] += result.feasible
                    sweeps[variant] += result.repair.sweeps
            records.append(
                {
                    "case": case.name,
                    "completed_per_policy": len(keys),
                    "repair_max_sweeps": variants[1].repair_max_sweeps,
                    "policies": [
                        {
                            "repair": policy,
                            "decode_seconds": times[index],
                            "median_decode_seconds": statistics.median(times[index]),
                            "total_completed_sweeps": sweeps[index],
                            "feasible": feasible[index],
                            "independent_errors": 0,
                        }
                        for index, policy in enumerate(("none", "continuous"))
                    ],
                    "claim": "Feature cost with changed assignments, not a general-engine speedup.",
                }
            )
        return records
