"""Baseline measurement of how often random keys decode to feasible model solutions.

Objective:
- Record, before any new decoder exists, the fraction of random keys the current model decoder
  turns into feasible solutions, and how good those solutions are (issue 125, phase 0).

Technical approach:
- Compile each fixture to a `CompiledLinearModel` with box-only and continuous-restoration
  policies, decode seeded uniform keys and check every result with the independent fixture checker.
- Report feasible count, gap to the known optimum on feasible keys and decode throughput.

Flow (pseudocode):
1. Build the baseline fixtures plus the structure fixtures, skipping the infeasible one.
2. For each policy, decode the same seeded keys and verify feasibility independently.
3. Summarize and optionally write one JSON file, refusing to overwrite.

Important criteria:
- Measures sampled-key feasibility and quality only; it makes no claim about search quality.
- Throughput is wall-clock on the host and is descriptive, never a pass/fail threshold.

Inputs/Outputs:
- Input: key count and seed.
- Output: one record per fixture with a summary per repair policy.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import statistics
import time
from .cases import PracticalCase
from .cases import PracticalModels
from .structure_cases import StructureCases
from pathlib import Path
from rko_solver.modeling.linear import CompiledLinearModel
from rko_solver.modeling.linear import LinearRow
from rko_solver.modeling.linear import LinearVariable
from rko_solver.modeling.records import ModelEvaluation
from typing import Any

POLICIES = ("none", "continuous")


class KeyFeasibilityStudy:
    """Measure sampled-key feasibility of the current decoder on fixed fixtures.

    Responsibilities:
    - Keep keys, policies and independent checking identical across fixtures.
    """

    @staticmethod
    def cases() -> tuple[PracticalCase, ...]:
        """Return fixtures with a known optimum, baseline first.

        Returns:
            tuple[PracticalCase, ...]: Baseline and structure fixtures.
        """
        every = PracticalModels.cases() + StructureCases.cases()
        return tuple(case for case in every if case.expected is not None)

    @staticmethod
    def _snapshot(case: PracticalCase, repair: str) -> CompiledLinearModel:
        """Compile one fixture under one repair policy."""
        return CompiledLinearModel(
            tuple(LinearVariable(v.name, v.name, v.lower, v.upper, v.kind) for v in case.variables),
            case.objective,
            tuple(LinearRow(row.name, row.coefficients, row.lower, row.upper) for row in case.rows),
            maximize=case.maximize,
            repair=repair,
        )

    @staticmethod
    def _gaps(case: PracticalCase, results: list[ModelEvaluation]) -> tuple[list[float], int]:
        """Collect relative gaps of feasible results and count checker disagreements.

        Args:
            case (PracticalCase): Fixture with a known optimum.
            results (list[ModelEvaluation]): Decoded results.

        Returns:
            tuple[list[float], int]: Gaps to the optimum (negative is better than the reference)
                and the number of results whose feasibility the independent checker disputes.
        """
        assert case.expected is not None
        scale = max(1.0, abs(case.expected))
        sign = -1.0 if case.maximize else 1.0
        gaps: list[float] = []
        errors = 0
        for result in results:
            values = {v.name: value for v, value in zip(case.variables, result.values, strict=True)}
            errors += result.feasible != PracticalModels.check(case, values)["feasible"]
            if result.feasible:
                gaps.append(sign * (result.objective - case.expected) / scale)
        return gaps, errors

    @staticmethod
    def measure(case: PracticalCase, repair: str, keys: int, seed: int) -> dict[str, Any]:
        """Decode seeded random keys and summarize feasibility and quality.

        Args:
            case (PracticalCase): Fixture with a known optimum.
            repair (str): `none` (box only) or `continuous` (restoration).
            keys (int): Number of uniform random key vectors.
            seed (int): Generator seed shared by every policy.

        Returns:
            dict[str, Any]: Counts, gaps on feasible keys, independent mismatches and throughput.

        Raises:
            ValueError: If the fixture has no known optimum.
        """
        if case.expected is None:
            raise ValueError("A known optimum is required to report gaps.")
        snapshot = KeyFeasibilityStudy._snapshot(case, repair)
        sample = np.random.default_rng(seed).random((keys, snapshot.n_keys))
        began = time.perf_counter()
        results = [snapshot.decode(row) for row in sample]
        elapsed = time.perf_counter() - began
        gaps, errors = KeyFeasibilityStudy._gaps(case, results)
        return {
            "repair": repair,
            "keys": keys,
            "feasible": len(gaps),
            "feasible_fraction": len(gaps) / keys if keys else 0.0,
            "best_gap": min(gaps, default=None),
            "median_gap": statistics.median(gaps) if gaps else None,
            "independent_errors": errors,
            "decode_seconds": elapsed,
            "decodes_per_second": keys / elapsed if elapsed > 0 else None,
        }

    @staticmethod
    def run(keys: int = 2048, seed: int = 2026) -> list[dict[str, Any]]:
        """Measure every fixture under both policies on the same keys.

        Args:
            keys (int): Random key vectors per fixture and policy.
            seed (int): Generator seed.

        Returns:
            list[dict[str, Any]]: One record per fixture.
        """
        return [
            {
                "case": case.name,
                "family": case.family,
                "policies": [
                    KeyFeasibilityStudy.measure(case, repair, keys, seed) for repair in POLICIES
                ],
            }
            for case in KeyFeasibilityStudy.cases()
        ]

    @staticmethod
    def write(path: Path, records: list[dict[str, Any]]) -> None:
        """Write records as JSON, never overwriting an existing file.

        Args:
            path (Path): Destination file.
            records (list[dict[str, Any]]): Study records.

        Raises:
            FileExistsError: If `path` already exists.
        """
        with path.open("x") as handle:
            json.dump(records, handle, indent=2)
            handle.write("\n")


def main() -> None:
    """Run the study from the command line and print or save the JSON records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    records = KeyFeasibilityStudy.run(args.keys, args.seed)
    if args.output is None:
        print(json.dumps(records, indent=2))
    else:
        KeyFeasibilityStudy.write(args.output, records)


if __name__ == "__main__":
    main()
