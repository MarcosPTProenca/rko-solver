"""Does the IPR algorithm improve best cost at equal wall time in a process-mode portfolio?

Objective:
- Compare the default four-algorithm portfolio with the same portfolio minus IPR, at the same
  wall-clock budget, on generated hard knapsacks, paired by seed (issue 136, quality step).

Technical approach:
- Run `RKO.from_model` in process mode for each arm and seed; arms differ only in the IPR entry.
- Record the best feasible objective and the completed evaluations of each arm.
- Summarize paired wins, ties and losses by seed, and the median objective per arm.

Flow (pseudocode):
1. For each fixture and seed, solve the arm with IPR and the arm without it, alternating order.
2. Record objectives and evaluation counts.
3. Report paired win, tie and loss counts and medians per fixture.

Important criteria:
- Equal budget is wall time for the whole solve, including process start-up in both arms.
- Arms run one after another, never concurrently, so they do not share cores.
- Timed cooperation is scheduling-dependent; the paired comparison, not one run, is the evidence.
- Benchmark only; the result does not change the default portfolio.

Inputs/Outputs:
- Input: fixture sizes, seeds, budget in seconds.
- Output: JSON with per-run records and a per-fixture summary.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from ..modeling.cases import PracticalModels
from ..modeling.hard_cases import HardCases
from pathlib import Path
from rko_solver import RKO

WITH_IPR = ("SA", "IPR", "BRKGA", "GA")
WITHOUT_IPR = ("SA", "BRKGA", "GA")
ARMS = {"with_ipr": WITH_IPR, "without_ipr": WITHOUT_IPR}


class IprContributionStudy:
    """Run the paired portfolio comparison and summarize it.

    Responsibilities:
    - Keep budget, fixture and seed identical across the two arms.
    """

    @staticmethod
    def solve(
        native: object, algorithms: tuple[str, ...], seed: int, seconds: float
    ) -> dict[str, object]:
        """Solve one arm in process mode and return its best objective and evaluations.

        Args:
            native (object): Native MathOpt model for the fixture.
            algorithms (tuple[str, ...]): Portfolio of this arm.
            seed (int): Search seed.
            seconds (float): Wall-clock budget for the whole solve.

        Returns:
            dict[str, object]: Objective (None if infeasible) and completed evaluations.
        """
        solver = RKO.from_model(
            native,
            algorithms=list(algorithms),
            adaptation="fixed",
            engine="python",
            execution="process",
            random_state=seed,
        )
        result = solver.solve(seconds)
        model = result.model
        assert model is not None
        feasible = model.best_feasible
        return {
            "objective": None if feasible is None else float(feasible.objective),
            "evaluations": sum(search.n_evaluations for search in result.searches),
        }

    @staticmethod
    def run(
        sizes: tuple[int, ...], rows: int, seeds: tuple[int, ...], seconds: float
    ) -> list[dict[str, object]]:
        """Run both arms for every fixture and seed, alternating which arm goes first.

        Args:
            sizes (tuple[int, ...]): Items per knapsack fixture.
            rows (int): Capacity rows per fixture.
            seeds (tuple[int, ...]): Generation and search seeds.
            seconds (float): Wall-clock budget per arm.

        Returns:
            list[dict[str, object]]: One record per arm, fixture and seed.
        """
        records = []
        for size in sizes:
            for seed in seeds:
                case = HardCases.build(size, rows, seed)
                native = PracticalModels.native(case, "mathopt")
                order = list(ARMS.items())
                if seed % 2:
                    order.reverse()
                for name, algorithms in order:
                    record = {"case": case.name, "seed": seed, "arm": name}
                    record.update(IprContributionStudy.solve(native, algorithms, seed, seconds))
                    records.append(record)
                    print(json.dumps(record), file=sys.stderr, flush=True)
        return records

    @staticmethod
    def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
        """Summarize paired differences (with IPR minus without) for each fixture.

        Args:
            records (list[dict[str, object]]): Raw records from `run`.

        Returns:
            list[dict[str, object]]: Mean and median paired difference, wins, ties, losses and the
            exact two-sided sign-test p-value over seeds where both arms were feasible.
        """
        summaries = []
        for case in sorted({str(r["case"]) for r in records}):
            diffs = IprContributionStudy._differences(records, case)
            summaries.append(IprContributionStudy._row(case, diffs))
        return summaries

    @staticmethod
    def _differences(records: list[dict[str, object]], case: str) -> list[float]:
        """Return with-minus-without differences for seeds where both arms are feasible."""
        arms: dict[int, dict[str, object]] = {}
        for record in records:
            if record["case"] == case:
                arms.setdefault(int(record["seed"]), {})[str(record["arm"])] = record["objective"]
        return [
            float(pair["with_ipr"]) - float(pair["without_ipr"])
            for pair in arms.values()
            if pair.get("with_ipr") is not None and pair.get("without_ipr") is not None
        ]

    @staticmethod
    def _row(case: str, diffs: list[float]) -> dict[str, object]:
        """Summarize one fixture's paired differences."""
        wins = sum(d > 1e-9 for d in diffs)
        losses = sum(d < -1e-9 for d in diffs)
        return {
            "case": case,
            "pairs": len(diffs),
            "mean_difference": statistics.fmean(diffs) if diffs else None,
            "median_difference": statistics.median(diffs) if diffs else None,
            "wins": wins,
            "losses": losses,
            "ties": len(diffs) - wins - losses,
            "sign_test_p": IprContributionStudy._sign_test(wins, losses),
        }

    @staticmethod
    def _sign_test(wins: int, losses: int) -> float | None:
        """Return the exact two-sided binomial sign-test p-value, ignoring ties."""
        n = wins + losses
        if n == 0:
            return None
        tail = sum(math.comb(n, k) for k in range(max(wins, losses), n + 1)) / 2**n
        return min(1.0, 2.0 * tail)


def main() -> None:
    """Run the comparison and write raw records plus the paired summary as JSON.

    Raises:
        ValueError: If a numeric argument is not positive.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=(100, 300))
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=tuple(range(1, 11)))
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.seconds <= 0 or arguments.rows <= 0:
        raise ValueError("Seconds and rows must be positive.")
    records = IprContributionStudy.run(
        tuple(arguments.sizes), arguments.rows, tuple(arguments.seeds), arguments.seconds
    )
    summary = IprContributionStudy.summarize(records)
    arguments.output.write_text(
        json.dumps({"records": records, "summary": summary}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
