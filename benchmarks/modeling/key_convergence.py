"""Baseline convergence of the current model decoder under real searches.

Objective:
- Record, before any new decoder exists, how fast SA, IPR and BRKGA reach feasible solutions and
  how good the best feasible one is at fixed evaluation budgets (issue 125, phase 0).

Technical approach:
- Build each fixture as a native MathOpt model and run `RKO.from_model` sequentially on the Python
  engine with `max_evaluations`, so each budget is an exact evaluation count, not wall-clock.
- Re-run fresh searches at each budget with the same seed; read the best feasible original
  objective and the time of the first feasible solution from the model result.

Flow (pseudocode):
1. For each fixture, algorithm, budget and seed, solve with that evaluation budget.
2. Convert the best feasible objective to a gap against the known optimum.
3. Summarize per fixture, algorithm and budget; optionally write the raw records.

Important criteria:
- Budgets are evaluation counts; times are descriptive wall-clock and never a threshold.
- One search per budget, so curves are not nested prefixes of a single run.

Inputs/Outputs:
- Input: fixture names, algorithms, budgets and seeds.
- Output: one raw record per run and a per-group summary.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from .cases import PracticalModels
from .key_feasibility import KeyFeasibilityStudy
from .schema import PracticalCase
from .structure_construction import StructureConstruction
from .violation_construction import ViolationConstruction
from pathlib import Path
from rko_solver import RKO
from typing import Any

ALGORITHMS = ("SA", "IPR", "BRKGA")
BUDGETS = (200, 800, 3200)
SEEDS = (0, 1, 2)
FRONTEND = "mathopt"


class KeyConvergenceStudy:
    """Run evaluation-budgeted searches on the fixtures and summarize them.

    Responsibilities:
    - Keep model, algorithm settings and seeds identical across fixtures.
    """

    @staticmethod
    def run(
        names: tuple[str, ...] | None = None,
        algorithms: tuple[str, ...] = ALGORITHMS,
        budgets: tuple[int, ...] = BUDGETS,
        seeds: tuple[int, ...] = SEEDS,
        progress: bool = False,
        decoder: str = "default",
    ) -> list[dict[str, Any]]:
        """Solve every fixture, algorithm, budget and seed once.

        Args:
            names (tuple[str, ...] | None): Fixture names, or None for every fixture.
            algorithms (tuple[str, ...]): Single-algorithm portfolios to run.
            budgets (tuple[int, ...]): Exact evaluation counts.
            seeds (tuple[int, ...]): Search seeds.
            progress (bool): Print one line per finished run to stderr.
            decoder (str): `default`, `structure` (L1) or `violation` (descent).

        Returns:
            list[dict[str, Any]]: One record per run.
        """
        cases = [c for c in KeyFeasibilityStudy.cases() if names is None or c.name in names]
        grid = list(itertools.product(cases, algorithms, budgets, seeds))
        records = []
        for done, (case, algorithm, budget, seed) in enumerate(grid, 1):
            records.append(KeyConvergenceStudy._one(case, algorithm, budget, seed, decoder))
            if progress:
                print(
                    f"{done}/{len(grid)} {case.name} {algorithm} {budget} {seed}", file=sys.stderr
                )
        return records

    @staticmethod
    def _construction(decoder: str) -> dict[str, object]:
        """Return the optional construction keyword for a decoder label."""
        constructions = {"structure": StructureConstruction, "violation": ViolationConstruction}
        return {"construction": constructions[decoder]()} if decoder in constructions else {}

    @staticmethod
    def _one(
        case: PracticalCase, algorithm: str, budget: int, seed: int, decoder: str
    ) -> dict[str, Any]:
        """Run one search and reduce its model result to a record."""
        solver = RKO.from_model(
            PracticalModels.native(case, FRONTEND),
            algorithms=[algorithm],
            adaptation="fixed",
            engine="python",
            execution="sequential",
            random_state=seed,
            max_evaluations=budget,
            **KeyConvergenceStudy._construction(decoder),
        )
        result = solver.solve(3600.0)
        model = result.model
        assert model is not None
        feasible = model.best_feasible
        return {
            "case": case.name,
            "decoder": decoder,
            "algorithm": algorithm,
            "budget": budget,
            "seed": seed,
            "evaluations": sum(search.n_evaluations for search in result.searches),
            "feasible": feasible is not None,
            "gap": None if feasible is None else KeyFeasibilityStudy.gap(case, feasible.objective),
            "seconds_to_first_feasible": model.time_to_first_feasible_seconds,
        }

    @staticmethod
    def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Aggregate raw records per fixture, algorithm and budget.

        Args:
            records (list[dict[str, Any]]): Raw run records.

        Returns:
            list[dict[str, Any]]: Run and feasible counts, gap and first-feasible-time statistics.
        """
        groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
        for record in records:
            key = (
                record["case"],
                record.get("decoder", "default"),
                record["algorithm"],
                record["budget"],
            )
            groups.setdefault(key, []).append(record)
        return [KeyConvergenceStudy._row(key, runs) for key, runs in groups.items()]

    @staticmethod
    def _row(key: tuple[str, str, str, int], runs: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize the runs of one fixture, algorithm and budget."""
        feasible = [r for r in runs if r["feasible"]]
        gaps = [r["gap"] for r in feasible]
        times = [r["seconds_to_first_feasible"] for r in feasible]
        return {
            "case": key[0],
            "decoder": key[1],
            "algorithm": key[2],
            "budget": key[3],
            "runs": len(runs),
            "feasible_runs": len(feasible),
            "best_gap": min(gaps, default=None),
            "median_gap": statistics.median(gaps) if gaps else None,
            "median_seconds_to_first_feasible": statistics.median(times) if times else None,
        }


def main() -> None:
    """Run the study from the command line; print the summary and save raw records if asked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=None, help="fixture name, repeatable")
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--decoder", choices=("default", "structure", "violation"), default="default"
    )
    parser.add_argument("--output", type=Path, default=None, help="raw records, never overwritten")
    args = parser.parse_args()
    records = KeyConvergenceStudy.run(
        tuple(args.case) if args.case else None,
        ALGORITHMS,
        tuple(args.budgets),
        tuple(args.seeds),
        progress=True,
        decoder=args.decoder,
    )
    if args.output is not None:
        KeyFeasibilityStudy.write(args.output, records)
    print(json.dumps(KeyConvergenceStudy.summarize(records), indent=2))


if __name__ == "__main__":
    main()
