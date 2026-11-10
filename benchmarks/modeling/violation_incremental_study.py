"""Compare full-rescan and incremental violation descent on generated linear models.

Objective:
- Measure whether sparse incremental activity updates preserve decoder behavior and improve cost.

Technical approach:
- Reproduce the merged full-rescan algorithm as a benchmark-only baseline.
- Compare it with the incremental implementation on identical deterministic key vectors or timed
  searches.
- Include model setup in search budgets and record actual evaluations and feasible objective gaps.

Flow (pseudocode):
1. Build generated qualification models and deterministic benchmark inputs.
2. Run both decoders on identical keys, or in paired order for equal-time search.
3. Emit raw JSON records and progress after each case or search.

Important criteria:
- This study code is benchmark-only and does not change the shipped solver.
- Generated families are synthetic stress tests, not representative industrial models.
- Decoder ordering alternates by seed to reduce systematic order bias.

Inputs/Outputs:
- Input: family, size, sample/search budget, seed and algorithm selections.
- Output: per-run JSON records with timing, evaluation count, feasibility and gap.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import statistics
import sys
import time
from .cases import PracticalModels
from .key_feasibility import KeyFeasibilityStudy
from .qualification_cases import FAMILIES
from .qualification_cases import QualificationCases
from .structure_construction import StructureConstruction
from .violation_construction import MAX_SWEEPS
from .violation_construction import TOLERANCE
from .violation_construction import ViolationConstruction
from pathlib import Path
from rko_solver import RKO
from rko_solver.modeling.construction import ConstructionInput


class FullRescanViolation:
    """Retain the merged full-row-rescan algorithm as a performance reference.

    Responsibilities:
    - Reproduce the pre-optimization greedy unit-move scoring for paired comparisons.
    """

    def __call__(self, keys: np.ndarray, context: ConstructionInput) -> list[float]:
        """Map keys and apply full-rescan violation descent.

        Args:
            keys (np.ndarray): Free-variable random keys.
            context (ConstructionInput): Immutable model construction context.

        Returns:
            list[float]: Original-order assignment after greedy descent.
        """
        values, _ = StructureConstruction._box(keys, context)
        movable = [
            index
            for index, (variable, domain) in enumerate(
                zip(context.variables, context.domains, strict=True)
            )
            if not domain.fixed and variable.domain != "continuous"
        ]
        current = ViolationConstruction._violation(values, context)
        for _ in range(MAX_SWEEPS):
            if current <= TOLERANCE:
                break
            best = self._best_full_move(values, movable, current, context)
            if best is None:
                break
            current, index, values[index] = best
        return values

    @staticmethod
    def _best_full_move(
        values: list[float], movable: list[int], current: float, context: ConstructionInput
    ) -> tuple[float, int, float] | None:
        """Score every unit move by recomputing all rows and return the best improving one."""
        best: tuple[float, int, float] | None = None
        for index, candidate in FullRescanViolation._moves(values, movable, context):
            trial = list(values)
            trial[index] = candidate
            score = ViolationConstruction._violation(trial, context)
            if score < current - TOLERANCE and (best is None or score < best[0]):
                best = (score, index, candidate)
        return best

    @staticmethod
    def _moves(
        values: list[float], movable: list[int], context: ConstructionInput
    ) -> list[tuple[int, float]]:
        """List in-domain unit moves in index order, then the -1 step before the +1 step."""
        moves = []
        for index in movable:
            domain = context.domains[index]
            for step in (-1.0, 1.0):
                candidate = values[index] + step
                if domain.lower <= candidate <= domain.upper:
                    moves.append((index, candidate))
        return moves


class ViolationIncrementalStudy:
    """Run deterministic paired decoder-speed and search-quality studies.

    Responsibilities:
    - Build matched records for the old full-rescan and incremental decoders.
    - Keep setup, objective gap, feasibility and completed evaluation counts explicit.
    """

    @staticmethod
    def decodes(samples: int, seed: int, sizes: tuple[int, ...]) -> list[dict[str, object]]:
        """Measure paired decode time and assignment agreement on all generated families.

        Args:
            samples (int): Number of deterministic key vectors per case.
            seed (int): Key generation seed.
            sizes (tuple[int, ...]): Generated model dimensions.

        Returns:
            list[dict[str, object]]: One timing and agreement record per family and size.
        """
        records = []
        for family in FAMILIES:
            for size in sizes:
                case, _ = QualificationCases.build(family, size, 20261107)
                keys = np.random.default_rng(seed).random((samples, len(case.variables)))
                old = StructureConstruction.bind(case, FullRescanViolation())
                incremental = StructureConstruction.bind(case, ViolationConstruction())
                old_seconds, old_results = ViolationIncrementalStudy._timed(old, keys)
                new_seconds, new_results = ViolationIncrementalStudy._timed(incremental, keys)
                records.append(
                    {
                        "family": case.name,
                        "samples": samples,
                        "seed": seed,
                        "full_rescan_seconds": old_seconds,
                        "incremental_seconds": new_seconds,
                        "speedup": old_seconds / new_seconds,
                        **ViolationIncrementalStudy._agreement(
                            case, old_results, new_results, old.context
                        ),
                    }
                )
                print(f"decode {case.name} complete", file=sys.stderr, flush=True)
        return records

    @staticmethod
    def _timed(model: object, keys: np.ndarray) -> tuple[float, list[object]]:
        """Decode every key once and return the elapsed seconds with the results."""
        begin = time.perf_counter()
        results = [model.decode(key) for key in keys]
        return time.perf_counter() - begin, results

    @staticmethod
    def _agreement(
        case: object,
        old_results: list[object],
        new_results: list[object],
        context: ConstructionInput,
    ) -> dict[str, object]:
        """Compare paired assignments by identity, objective, violation and feasibility."""
        pairs = list(zip(old_results, new_results, strict=True))
        return {
            "identical_assignments": sum(np.array_equal(a.values, b.values) for a, b in pairs),
            "max_objective_delta": max(
                abs(
                    ViolationIncrementalStudy._objective(case, a)
                    - ViolationIncrementalStudy._objective(case, b)
                )
                for a, b in pairs
            ),
            "max_violation_delta": max(
                abs(
                    ViolationConstruction._violation(list(a.values), context)
                    - ViolationConstruction._violation(list(b.values), context)
                )
                for a, b in pairs
            ),
            "feasible_full_rescan": sum(a.feasible for a, _ in pairs),
            "feasible_incremental": sum(b.feasible for _, b in pairs),
        }

    @staticmethod
    def _objective(case: object, result: object) -> float:
        """Return the original objective of one decoded assignment."""
        return sum(c * x for c, x in zip(case.objective, result.values, strict=True))

    @staticmethod
    def searches(
        seconds: float,
        seeds: tuple[int, ...],
        sizes: tuple[int, ...],
        algorithms: tuple[str, ...],
        families: tuple[str, ...],
    ) -> list[dict[str, object]]:
        """Compare both decoders under matched total wall-clock budgets.

        Args:
            seconds (float): Per-run total wall-clock budget including model setup.
            seeds (tuple[int, ...]): Independent search seeds.
            sizes (tuple[int, ...]): Generated model dimensions.
            algorithms (tuple[str, ...]): Search algorithms tested independently.
            families (tuple[str, ...]): Generated model families to include.

        Returns:
            list[dict[str, object]]: One run record per case, algorithm, seed and decoder.
        """
        records = []
        for family in families:
            for size in sizes:
                case, _ = QualificationCases.build(family, size, 20261107)
                native = PracticalModels.native(case, "mathopt")
                for algorithm in algorithms:
                    for seed in seeds:
                        decoders = ("full_rescan", "incremental")
                        if seed % 2:
                            decoders = tuple(reversed(decoders))
                        for decoder in decoders:
                            record = ViolationIncrementalStudy._one_search(
                                case, native, algorithm, seed, decoder, seconds
                            )
                            records.append(record)
                            print(
                                f"search {case.name} {algorithm} seed={seed} {decoder}: "
                                f"{record['evaluations']} evals, gap={record['gap']}",
                                file=sys.stderr,
                                flush=True,
                            )
        return records

    @staticmethod
    def _one_search(
        case: object, native: object, algorithm: str, seed: int, decoder: str, seconds: float
    ) -> dict[str, object]:
        """Run one search with the named decoder and reduce it to a record."""
        constructor = FullRescanViolation() if decoder == "full_rescan" else ViolationConstruction()
        started = time.perf_counter()
        solver = RKO.from_model(
            native,
            construction=constructor,
            algorithms=[algorithm],
            adaptation="fixed",
            engine="python",
            execution="sequential",
            random_state=seed,
        )
        result = solver.solve(max(0.05, seconds - (time.perf_counter() - started)))
        elapsed = time.perf_counter() - started
        model = result.model
        assert model is not None
        feasible = model.best_feasible
        return {
            "case": case.name,
            "algorithm": algorithm,
            "seed": seed,
            "decoder": decoder,
            "budget_seconds": seconds,
            "elapsed_seconds": elapsed,
            "evaluations": sum(search.n_evaluations for search in result.searches),
            "feasible": feasible is not None,
            "gap": None if feasible is None else KeyFeasibilityStudy.gap(case, feasible.objective),
        }

    @staticmethod
    def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
        """Aggregate paired search records by case, algorithm and decoder.

        Args:
            records (list[dict[str, object]]): Raw search records.

        Returns:
            list[dict[str, object]]: Feasible rate, median gap and median evaluations.
        """
        groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
        for record in records:
            key = (str(record["case"]), str(record["algorithm"]), str(record["decoder"]))
            groups.setdefault(key, []).append(record)
        return [ViolationIncrementalStudy._row(key, runs) for key, runs in groups.items()]

    @staticmethod
    def _row(key: tuple[str, str, str], runs: list[dict[str, object]]) -> dict[str, object]:
        """Summarize the runs of one case, algorithm and decoder."""
        feasible = [run for run in runs if run["feasible"]]
        gaps = [float(run["gap"]) for run in feasible]
        return {
            "case": key[0],
            "algorithm": key[1],
            "decoder": key[2],
            "runs": len(runs),
            "feasible_runs": len(feasible),
            "median_gap": statistics.median(gaps) if gaps else None,
            "median_evaluations": statistics.median(int(run["evaluations"]) for run in runs),
        }


def main() -> None:
    """Run decode or equal-wall-time experiments and write raw plus summary JSON.

    Raises:
        ValueError: If the requested mode or budget values are invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("decode", "search"), required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+", default=(48, 192))
    parser.add_argument("--algorithms", nargs="+", default=("SA", "BRKGA"))
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.samples <= 0 or arguments.seconds <= 0 or arguments.seeds <= 0:
        raise ValueError("Samples, seconds and seeds must be positive.")
    if arguments.mode == "decode":
        result = {
            "records": ViolationIncrementalStudy.decodes(
                arguments.samples, 2026, tuple(arguments.sizes)
            )
        }
    else:
        records = ViolationIncrementalStudy.searches(
            arguments.seconds,
            tuple(range(arguments.seeds)),
            tuple(arguments.sizes),
            tuple(arguments.algorithms),
            tuple(arguments.families),
        )
        result = {"records": records, "summary": ViolationIncrementalStudy.summarize(records)}
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
