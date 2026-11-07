"""Fixed-key diversity and paired total-wall-time qualification of model decoders.

Objective:
- Measure whether benchmark-only construction helps larger algebraic models at equal total time.

Technical approach:
- Validate sampled assignments with original fixture arithmetic and report exact output collisions.
- Time native model creation, conversion and compilation inside each arm's budget; solve gets only
  the remainder. Record overruns instead of pretending cooperative stopping is a hard deadline.
- Run arms sequentially, alternate order, and retain every raw result in exclusive JSONL output.

Flow (pseudocode):
1. Reserve output, identify sources/inputs and warm optional imports and algorithm paths.
2. Independently certify references, then probe identical keys for each decoder.
3. Run paired searches and flush each result; abort on errors without deleting partial evidence.

Important criteria:
- Sample-output diversity is not evolutionary-population diversity.
- No reference witness or solution is injected into searches or repair.
- Synthetic stress tests and a single host cannot qualify arbitrary models or prove speedups.

Inputs/Outputs:
- Input: cases, total seconds per arm, paired seeds, fixed-key sample count and output path.
- Output: metadata, reference, probe and search records, followed by a completion marker.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import numpy as np
import os
import platform
import subprocess
import sys
import time
from .cases import PracticalModels
from .key_feasibility import KeyFeasibilityStudy
from .qualification_cases import FAMILIES
from .qualification_cases import QualificationCases
from .schema import PracticalCase
from .structure_construction import StructureConstruction
from pathlib import Path
from rko_solver import RKO
from rko_solver.modeling.records import ModelEvaluation
from typing import Any
from typing import TextIO


class DecoderQualification:
    """Compare default and prototype construction without changing the shipped decoder.

    Responsibilities:
    - Enforce independent validity and explicit setup-inclusive timing.
    - Keep incomplete campaigns and all paired outcomes visible.
    """

    @staticmethod
    def arm_order(position: int) -> tuple[str, str]:
        """Return alternating decoder order for a paired seed position."""
        return ("default", "structure") if position % 2 == 0 else ("structure", "default")

    @staticmethod
    def validate(case: PracticalCase, evaluation: ModelEvaluation) -> None:
        """Require objective and feasibility agreement with original fixture data.

        Args:
            case (PracticalCase): Original data, independent of compiled arrays.
            evaluation (ModelEvaluation): Completed generated-model result.

        Raises:
            RuntimeError: If independent objective or feasibility disagrees.
        """
        values = dict(zip((v.name for v in case.variables), evaluation.values, strict=True))
        checked = PracticalModels.check(case, values)
        same_objective = math.isclose(
            checked["objective"], evaluation.objective, rel_tol=1e-12, abs_tol=1e-9
        )
        if checked["feasible"] != evaluation.feasible or not same_objective:
            raise RuntimeError(f"Independent model disagreement: {case.name}")

    @staticmethod
    def probe(case: PracticalCase, decoder: str, keys: np.ndarray) -> dict[str, Any]:
        """Decode fixed keys and measure exact assignment collisions and adjacent distances.

        Args:
            case (PracticalCase): Original model data.
            decoder (str): Default mapping or structure prototype.
            keys (np.ndarray): Shared random-key sample, one chromosome per row.

        Returns:
            dict[str, Any]: Independent feasibility, exact output diversity and elapsed time.
        """
        policy = "structure" if decoder == "structure" else "continuous"
        snapshot = KeyFeasibilityStudy._snapshot(case, policy)
        started = time.perf_counter()
        results = [snapshot.decode(row) for row in keys]
        elapsed = time.perf_counter() - started
        for result in results:
            DecoderQualification.validate(case, result)
        diversity = DecoderQualification._diversity(case, results)
        return {
            "type": "probe",
            "case": case.name,
            "decoder": decoder,
            "samples": len(keys),
            "feasible_samples": sum(r.feasible for r in results),
            "decode_seconds": elapsed,
            **diversity,
        }

    @staticmethod
    def _diversity(case: PracticalCase, results: list[ModelEvaluation]) -> dict[str, Any]:
        """Summarize exact decoded outputs, with domain-normalized adjacent distances."""
        values = np.array([r.values for r in results], dtype=float)
        values[values == 0] = 0.0
        widths = np.array([max(1.0, v.upper - v.lower) for v in case.variables])
        distance = np.abs(np.diff(values, axis=0)) / widths
        count = len(np.unique(values, axis=0))
        feasible = {r.values for r in results if r.feasible}
        return {
            "unique_assignments": count,
            "unique_feasible_assignments": len(feasible),
            "collision_fraction": 1 - count / len(results),
            "mean_adjacent_distance": float(distance.sum() / max(1, distance.size)),
            "assignment_digest": hashlib.sha256(values.tobytes()).hexdigest(),
        }

    @staticmethod
    def timed(
        case: PracticalCase, algorithm: str, decoder: str, seed: int, seconds: float
    ) -> dict[str, Any]:
        """Run a public factory/search arm with setup charged against total time.

        Args:
            case (PracticalCase): Original model data.
            algorithm (str): One algorithm, not a portfolio sharing budgets.
            decoder (str): Default or benchmark-only structure constructor.
            seed (int): Paired search seed.
            seconds (float): Total target seconds, including setup, excluding result checking.

        Returns:
            dict[str, Any]: Timing reconciliation, completed work and independently checked quality.
        """
        started = time.perf_counter()
        solver = RKO.from_model(
            PracticalModels.native(case, "mathopt"),
            algorithms=[algorithm],
            adaptation="fixed",
            engine="python",
            execution="sequential",
            prefill_pool=False,
            random_state=seed,
            **({"construction": StructureConstruction()} if decoder == "structure" else {}),
        )
        prepared = time.perf_counter()
        record = DecoderQualification._empty(case, algorithm, decoder, seed, seconds)
        setup = prepared - started
        if setup >= seconds:
            record.update(
                status="setup_exhausted",
                setup_seconds=setup,
                total_seconds=setup,
                overrun_seconds=setup - seconds,
            )
            return record
        result = solver.solve(seconds - setup)
        finished = time.perf_counter()
        record.update(
            setup_seconds=setup,
            solve_seconds=finished - prepared,
            total_seconds=finished - started,
            overrun_seconds=max(0.0, finished - started - seconds),
            evaluations=sum(s.n_evaluations for s in result.searches),
        )
        assert result.model is not None
        DecoderQualification.validate(case, result.model.search_incumbent.evaluation)
        best = result.model.best_feasible
        if best is not None:
            DecoderQualification.validate(case, best.evaluation)
            record.update(
                feasible=True,
                objective=best.objective,
                gap=KeyFeasibilityStudy.gap(case, best.objective),
            )
            first = result.model.time_to_first_feasible_seconds
            record["seconds_to_first_feasible"] = None if first is None else setup + first
        return record

    @staticmethod
    def _empty(
        case: PracticalCase, algorithm: str, decoder: str, seed: int, seconds: float
    ) -> dict[str, Any]:
        """Initialize a complete record, including a setup-exhausted or no-feasible outcome."""
        return {
            "type": "search",
            "case": case.name,
            "algorithm": algorithm,
            "decoder": decoder,
            "seed": seed,
            "budget_seconds": seconds,
            "status": "completed",
            "feasible": False,
            "objective": None,
            "gap": None,
            "seconds_to_first_feasible": None,
            "setup_seconds": 0.0,
            "solve_seconds": 0.0,
            "total_seconds": 0.0,
            "overrun_seconds": 0.0,
            "evaluations": 0,
        }

    @staticmethod
    def metadata(cases: tuple[PracticalCase, ...], settings: dict[str, Any]) -> dict[str, Any]:
        """Identify actual source files, fixtures and runtime, not just the base revision."""
        root = Path(__file__).resolve().parents[2]
        sources = sorted((root / "src/rko_solver").rglob("*.py"))
        sources += sorted((root / "benchmarks/modeling").glob("*.py"))
        sources += sorted((root / "src/rko_solver/parameters").glob("*.toml"))
        sources += [root / "pyproject.toml", root / "uv.lock"]
        return {
            "type": "metadata",
            "settings": settings,
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "sources": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
            "inputs": {c.name: hashlib.sha256(repr(c).encode()).hexdigest() for c in cases},
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpus": os.process_cpu_count(),
            "metric_limits": "Exact sampled-output collisions, not evolving-population diversity; "
            "cooperative timing, not hard deadlines; synthetic cases only.",
        }

    @staticmethod
    def reference(case: PracticalCase) -> dict[str, Any]:
        """Require separate HiGHS certification, without injecting its solution."""
        result, elapsed = PracticalModels._solve_reference(case)
        if result.status != 0:
            raise RuntimeError(f"Reference not certified for {case.name}: {result.message}")
        objective = float(-result.fun if case.maximize else result.fun)
        if not math.isclose(objective, case.expected, rel_tol=1e-9, abs_tol=1e-7):
            raise RuntimeError(f"Reference optimum disagrees for {case.name}")
        return {
            "type": "reference",
            "case": case.name,
            "status": "certified_optimal",
            "objective": objective,
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def warmup() -> None:
        """Warm optional imports and both algorithm/decoder paths outside measurement."""
        case, _ = QualificationCases.build("ranged_integer", 6, 0)
        for algorithm, decoder in itertools.product(("SA", "BRKGA"), ("default", "structure")):
            solver = RKO.from_model(
                PracticalModels.native(case, "mathopt"),
                algorithms=[algorithm],
                adaptation="fixed",
                execution="sequential",
                engine="python",
                max_evaluations=1,
                **({"construction": StructureConstruction()} if decoder == "structure" else {}),
            )
            solver.solve(5.0)

    @staticmethod
    def _write(handle: TextIO, record: dict[str, Any]) -> None:
        """Flush one strict JSON record and report progress without retaining the whole campaign."""
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
        print(
            f"{record['type']} {record.get('case', '')} {record.get('decoder', '')} "
            f"{record.get('algorithm', '')} {record.get('seed', '')}",
            file=sys.stderr,
            flush=True,
        )

    @staticmethod
    def run(
        output: Path,
        cases: tuple[PracticalCase, ...],
        seconds: float = 2.0,
        seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
        samples: int = 128,
    ) -> None:
        """Reserve output and execute the entire paired campaign, failing on the first error.

        Args:
            output (Path): Exclusive append-only JSONL destination.
            cases (tuple[PracticalCase, ...]): Frozen original fixtures with known optima.
            seconds (float): Total target time per search arm.
            seeds (tuple[int, ...]): Paired seeds; order alternates by their position.
            samples (int): Fixed-key vectors per case and decoder.

        Raises:
            FileExistsError: If output already exists.
            ValueError: If the campaign configuration is invalid.
            Exception: Reference, conversion, search or independent-validation failure.
        """
        if not math.isfinite(seconds) or seconds <= 0 or samples < 1 or not seeds or not cases:
            raise ValueError("Positive time/samples and nonempty cases/seeds are required.")
        settings = {
            "seconds": seconds,
            "seeds": seeds,
            "samples": samples,
            "key_seed": 20261107,
            "algorithms": ["SA", "BRKGA"],
            "arm_order": "alternating by seed position",
            "warmup": "both algorithm/decoder paths before measurement",
        }
        with output.open("x") as handle:
            DecoderQualification._write(handle, DecoderQualification.metadata(cases, settings))
            try:
                DecoderQualification.warmup()
                DecoderQualification._campaign(handle, cases, settings)
            except Exception as error:
                DecoderQualification._write(handle, {"type": "error", "message": str(error)})
                raise
            DecoderQualification._write(handle, {"type": "complete"})

    @staticmethod
    def _campaign(
        handle: TextIO, cases: tuple[PracticalCase, ...], settings: dict[str, Any]
    ) -> None:
        """Run references and fixed-key probes, followed by sequential paired searches."""
        for case in cases:
            DecoderQualification._write(handle, DecoderQualification.reference(case))
            keys = np.random.default_rng(settings["key_seed"]).random(
                (settings["samples"], len(case.variables))
            )
            for decoder in ("default", "structure"):
                DecoderQualification._write(handle, DecoderQualification.probe(case, decoder, keys))
        grid = itertools.product(cases, settings["algorithms"], enumerate(settings["seeds"]))
        for case, algorithm, (position, seed) in grid:
            for decoder in DecoderQualification.arm_order(position):
                record = DecoderQualification.timed(
                    case, algorithm, decoder, seed, settings["seconds"]
                )
                record["pair_position"] = position
                DecoderQualification._write(handle, record)


def main() -> None:
    """Run the frozen six-fixture qualification campaign, writing incremental evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    cases = tuple(
        QualificationCases.build(family, size, 20261107)[0]
        for family, size in itertools.product(FAMILIES, (48, 192))
    )
    DecoderQualification.run(args.output, cases, args.seconds, tuple(args.seeds), args.samples)


if __name__ == "__main__":
    main()
