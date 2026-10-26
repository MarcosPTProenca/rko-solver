"""Seeded practical native-model acceptance campaign with independent reference evidence.

Objective:
- Measure actual feasibility and original-objective quality across supported native libraries.

Technical approach:
- Reuse the same cited input, BRKGA settings, seeds and budgets across all frontends.
- Test feasibility-first and one explicit illustrative penalty weight without per-case tuning.
- Keep reference optimization, model construction, compilation and independent validation separate.
- Persist every run, including no-feasible-found and error outcomes, in strict JSONL.

Flow (pseudocode):
1. Resolve separately timed reference answers and freeze input/runtime/source provenance.
2. Build a real native model, convert it and solve through the public RKO factory.
3. Independently check returned witnesses and preserve original search-incumbent identity.
4. Write every outcome and summarize all seeds without dropping failures.

Important criteria:
- Reference witnesses never seed, repair or rerank RKO search.
- Signed gaps use exact-model reference objectives; feasibility uses declared tolerances.
- This local acceptance campaign is not a solver-dominance or engine-speedup benchmark.
- Output creation is exclusive to avoid overwriting prior measurements.

Inputs/Outputs:
- Input: optional frontend extras, benchmark-only reference dependency, seeds and time budget.
- Output: reproducible strict JSONL observations and per-case/frontend/scoring summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from .modeling_cases import PracticalCase
from .modeling_cases import PracticalModels
from .modeling_provenance import CampaignProvenance
from .modeling_summary import CampaignSummary
from dataclasses import asdict
from datetime import UTC
from datetime import datetime
from numbers import Real
from pathlib import Path
from rko_solver import RKO
from rko_solver.modeling.results import ModelResult
from rko_solver.modeling.results import ModelSolution
from rko_solver.solver import RKOResult
from typing import Any
from typing import Required
from typing import TypedDict
from typing import Unpack


class ObservationOptions(TypedDict, total=False):
    """Describe required timing/oracle data and optional common restoration settings.

    Responsibilities:
    - Keep observation policy explicit without positional argument growth.

    Attributes:
        time_limit (float): Required per-search duration.
        reference (dict): Required independent original-model oracle.
        repair (str): Common restoration policy, never selected per case.
        repair_max_sweeps (int): Common deterministic sweep cap.
    """

    time_limit: Required[float]
    reference: Required[dict[str, Any]]
    repair: str
    repair_max_sweeps: int


class ModelValidationCampaign:
    """Run and summarize bounded practical model acceptance without changing search callbacks.

    Responsibilities:
    - Keep validation/reference arithmetic outside measured search and record every outcome.
    - Tie observations to exact source/input/runtime identities, including dirty working trees.

    Attributes:
        frontends (tuple[str, ...]): All initial supported native libraries.
        penalty_weight (float): Illustrative raw-unit weight, not a universal recommendation.
    """

    frontends = ("pyomo", "pulp", "mathopt")
    penalty_weight = 1000.0

    @staticmethod
    def definition(case: PracticalCase) -> dict[str, Any]:
        """Export original case data without nonstandard JSON infinities or compiled dependencies.

        Args:
            case (PracticalCase): Offline cited formulation.

        Returns:
            dict[str, Any]: Complete mathematical input and a stable fixture SHA256.
        """
        data = {
            "name": case.name,
            "family": case.family,
            "source": case.source,
            "adaptation": case.adaptation,
            "variables": [asdict(v) for v in case.variables],
            "objective": list(case.objective),
            "maximize": case.maximize,
            "expected_objective": case.expected,
            "rows": [
                {
                    "name": row.name,
                    "coefficients": row.coefficients,
                    "lower": row.lower if math.isfinite(row.lower) else None,
                    "upper": row.upper if math.isfinite(row.upper) else None,
                }
                for row in case.rows
            ],
        }
        data["fixture_sha256"] = hashlib.sha256(
            json.dumps(data, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        return data

    @classmethod
    def metadata(
        cls,
        cases: tuple[PracticalCase, ...],
        seeds: int,
        time_limit: float,
        repair: str = "continuous",
        repair_max_sweeps: int = 1024,
    ) -> dict[str, Any]:
        """Capture source and runtime provenance without treating dirty HEAD as executed source.

        Args:
            cases (tuple[PracticalCase, ...]): Entire campaign, including inconsistent stress cases.
            seeds (int): Number of seeds starting at zero.
            time_limit (float): Seconds for each single-algorithm serial search.
            repair (str): Explicit common mapping/restoration policy.
            repair_max_sweeps (int): Explicit common restoration work cap.

        Returns:
            dict[str, Any]: Runtime versions, source hashes, settings and complete case inputs.

        Raises:
            OSError: If benchmark/production source cannot be read.
            subprocess.CalledProcessError: If repository identity cannot be inspected.
        """
        root = Path(__file__).resolve().parents[1]
        sources = [
            "benchmarks/modeling_cases.py",
            "benchmarks/modeling_schema.py",
            "benchmarks/modeling_provenance.py",
            "benchmarks/modeling_summary.py",
            "benchmarks/modeling_validation.py",
            "benchmarks/knapsack_utils.py",
            "pyproject.toml",
            "uv.lock",
        ]
        sources.extend(
            path.relative_to(root).as_posix()
            for path in sorted((root / "src/rko_solver").rglob("*.py"))
            if "tests" not in path.relative_to(root / "src/rko_solver").parts
        )
        sources.extend(
            path.relative_to(root).as_posix()
            for path in sorted((root / "src/rko_solver/parameters").glob("*.toml"))
        )
        runtime = CampaignProvenance.runtime()
        return {
            "kind": "metadata",
            "schema_version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
            ),
            "source_sha256": {
                path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in sources
            },
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_model": runtime["cpu_model"],
            "python": platform.python_version(),
            "available_cpus": os.process_cpu_count(),
            "versions": runtime["versions"],
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "algorithm": "BRKGA",
            "adaptation": "fixed",
            "engine": "python",
            "execution": "sequential",
            "prefill_pool": False,
            "seeds": list(range(seeds)),
            "time_limit_per_search": time_limit,
            "frontends": list(cls.frontends),
            "absolute_tolerance": 1e-06,
            "relative_tolerance": 0.0,
            "repair": repair,
            "repair_max_sweeps": repair_max_sweeps,
            "scoring_modes": ["feasibility_first", "weighted_penalty"],
            "illustrative_penalty_weight": cls.penalty_weight,
            "cases": [cls.definition(case) for case in cases],
            "claims": (
                "Local small public-model acceptance only; not industrial certification, "
                "not a generic engine speedup, not superiority over mathematical solvers."
            ),
        }

    @staticmethod
    def validate_solution(case: PracticalCase, solution: ModelSolution) -> dict[str, Any]:
        """Check original fixture semantics independently of generated numerical callbacks.

        Args:
            case (PracticalCase): Original coefficients/domains.
            solution (ModelSolution): Completed model/search witness, feasible or diagnostic.

        Returns:
            dict[str, Any]: Independent original objective/domain/constraint checks.

        Raises:
            AssertionError: If decoder semantics, feasibility or original objective disagree.
            ValueError: If assignment coordinates are malformed.
        """
        values = {assignment.label: assignment.value for assignment in solution.assignments}
        check = PracticalModels.check(case, values)
        if check["feasible"] != solution.feasible or not math.isclose(
            check["objective"], solution.objective, rel_tol=1e-12, abs_tol=1e-08
        ):
            raise AssertionError(
                f"Independent original-model validation failed for {case.name}: {check}"
            )
        return check

    @classmethod
    def observe(
        cls,
        case: PracticalCase,
        frontend: str,
        scoring: str,
        seed: int,
        **options: Unpack[ObservationOptions],
    ) -> dict[str, Any]:
        """Construct, solve and independently validate one unmodified native-model run.

        Args:
            case (PracticalCase): Offline cited formulation.
            frontend (str): Qualified native frontend identifier.
            scoring (str): Feasibility-first or weighted-penalty mode.
            seed (int): Existing RKO seed source.
            options (ObservationOptions): Required per-search duration and independent
                reference, plus optional common restoration policy and sweep cap.

        Returns:
            dict[str, Any]: Quality, counts, timing and independent validation evidence.

        Raises:
            AssertionError: If returned model evidence or source identity is inconsistent.
            ValueError: If native model data/settings are invalid.
            ImportError: If an optional frontend is unavailable.
            RuntimeError: If ordinary RKO execution fails.
        """
        time_limit = options["time_limit"]
        reference = options["reference"]
        repair = options.get("repair", "continuous")
        repair_max_sweeps = options.get("repair_max_sweeps", 1024)
        started = time.perf_counter()
        native = PracticalModels.native(case, frontend)
        build_seconds = time.perf_counter() - started
        weight = cls.penalty_weight if scoring == "weighted_penalty" else None
        solver = RKO.from_model(
            native,
            engine="python",
            algorithms=["BRKGA"],
            adaptation="fixed",
            execution="sequential",
            random_state=seed,
            prefill_pool=False,
            scoring=scoring,
            penalty_weight=weight,
            absolute_tolerance=1e-06,
            relative_tolerance=0.0,
            repair=repair,
            repair_max_sweeps=repair_max_sweeps,
        )
        result = solver.solve(time_limit=time_limit)
        model = result.model
        assert model is not None and model.search_incumbent.keys == result.keys
        assert model.search_incumbent.search_score == result.cost
        checked_at = time.perf_counter()
        incumbent_check = cls.validate_solution(case, model.search_incumbent)
        witness = model.best_feasible
        witness_check = cls.validate_solution(case, witness) if witness is not None else None
        cls._unchanged(model, native, frontend)
        cls._witness(model, result, scoring, reference, witness_check)
        gap = cls._gap(case, witness, reference)
        return {
            "kind": "run",
            "case": case.name,
            "family": case.family,
            "frontend": frontend,
            "maximize": case.maximize,
            "scoring": scoring,
            "penalty_weight": weight,
            "repair": repair,
            "repair_max_sweeps": repair_max_sweeps,
            "incumbent_repair": asdict(model.search_incumbent.repair),
            "witness_repair": asdict(witness.repair) if witness is not None else None,
            "seed": seed,
            "status": model.status,
            "n_variables": len(case.variables),
            "n_keys": solver.n_variables,
            "n_rows": len(case.rows),
            "nnz": sum(len(row.coefficients) for row in case.rows),
            "search_score": result.cost,
            "search_incumbent_feasible": model.search_incumbent.feasible,
            "search_incumbent_objective_diagnostic": model.search_incumbent.objective,
            "best_feasible_objective": witness.objective if witness is not None else None,
            "separate_feasible_witness": witness is not None and witness.keys != result.keys,
            "reference_status": reference["status"],
            "reference_objective": reference["objective"],
            "signed_gap_to_exact_reference_percent": gap,
            "n_evaluations": result.n_evaluations,
            "native_build_seconds": build_seconds,
            "compilation_seconds": model.compilation_seconds,
            "solve_seconds": result.elapsed_seconds,
            "time_to_first_feasible_seconds": model.time_to_first_feasible_seconds,
            "independent_validation_seconds": time.perf_counter() - checked_at,
            "independent_search_incumbent": incumbent_check,
            "independent_feasible_witness": witness_check,
            "source_fingerprint": model.source.fingerprint,
            "source_unchanged": True,
            "stop_reason": result.stop_reason,
        }

    @staticmethod
    def summary(path: Path) -> list[dict[str, Any]]:
        """Aggregate every observed seed, retaining errors and no-feasible outcomes in denominators.

        Args:
            path (Path): Campaign JSONL observations.

        Returns:
            list[dict[str, Any]]: Per-case/frontend/mode feasibility, gap and timing summaries.

        Raises:
            OSError: If observations cannot be read.
            ValueError: If JSONL is malformed.
        """
        return CampaignSummary.read(path)

    @classmethod
    def run(
        cls,
        output: Path,
        *,
        seeds: int = 10,
        time_limit: float = 0.5,
        repair: str = "continuous",
        repair_max_sweeps: int = 1024,
    ) -> list[dict[str, Any]]:
        """Write all planned runs and fail loudly if any independent semantic check errors.

        Args:
            output (Path): Exclusive JSONL output path; prior measurements cannot be overwritten.
            seeds (int): Positive seed count starting at zero.
            time_limit (float): Positive finite single-search allowance in seconds.
            repair (str): Common restoration policy, never selected per case.
            repair_max_sweeps (int): Common deterministic work cap.

        Returns:
            list[dict[str, Any]]: Per-case/frontend/scoring summaries including every outcome.

        Raises:
            ValueError: If seed count or search allowance is invalid.
            FileExistsError: If output already exists.
            ImportError: If benchmark-only reference dependencies are unavailable.
            RuntimeError: If a reference or any recorded run fails independent validation/execution.
            OSError: If source/input/output files cannot be accessed.
        """
        cls._search_settings(seeds, time_limit)
        cls._repair_settings(repair, repair_max_sweeps)
        if output.exists():
            raise FileExistsError(output)
        cases = PracticalModels.cases()
        references = {case.name: PracticalModels.reference(case) for case in cases}
        metadata = cls.metadata(cases, seeds, time_limit, repair, repair_max_sweeps)
        metadata["references"] = references
        output.parent.mkdir(parents=True, exist_ok=True)
        errors = 0
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, allow_nan=False) + "\n")
            for case in cases:
                for frontend in cls.frontends:
                    for scoring in ("feasibility_first", "weighted_penalty"):
                        for seed in range(seeds):
                            try:
                                record = cls.observe(
                                    case,
                                    frontend,
                                    scoring,
                                    seed,
                                    time_limit=time_limit,
                                    reference=references[case.name],
                                    repair=repair,
                                    repair_max_sweeps=repair_max_sweeps,
                                )
                            except Exception as exc:
                                errors += 1
                                record = {
                                    "kind": "run",
                                    "case": case.name,
                                    "family": case.family,
                                    "frontend": frontend,
                                    "scoring": scoring,
                                    "seed": seed,
                                    "maximize": case.maximize,
                                    "status": "error",
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
                            stream.flush()
        summaries = cls.summary(output)
        if errors:
            raise RuntimeError(
                f"{errors} campaign runs failed; every error is preserved in {output}."
            )
        return summaries

    @classmethod
    def main(cls) -> None:
        """Run the reproducible campaign or print summaries from existing observations.

        Raises:
            ValueError: If command-line settings or stored JSON are invalid.
            OSError: If source/input/output files cannot be accessed.
            ImportError: If optional native/reference runtimes are unavailable.
            RuntimeError: If reference validation or a campaign run fails.
        """
        parser = argparse.ArgumentParser(description="Validate practical native LP/IP/MILP models.")
        parser.add_argument("--seeds", type=int, default=10)
        parser.add_argument("--time-limit", type=float, default=0.5)
        parser.add_argument("--repair", choices=("none", "continuous"), default="continuous")
        parser.add_argument("--repair-max-sweeps", type=int, default=1024)
        parser.add_argument(
            "--output", type=Path, default=Path("docs/audits/modeling-validation.jsonl")
        )
        parser.add_argument("--summarize", type=Path)
        args = parser.parse_args()
        summaries = (
            cls.summary(args.summarize)
            if args.summarize is not None
            else cls.run(
                args.output,
                seeds=args.seeds,
                time_limit=args.time_limit,
                repair=args.repair,
                repair_max_sweeps=args.repair_max_sweeps,
            )
        )
        print(json.dumps(summaries, indent=2, allow_nan=False))

    @staticmethod
    def _unchanged(model: ModelResult, native: object, frontend: str) -> None:
        """Verify that solving did not initialize or mutate original native assignments.

        Args:
            model (ModelResult): Completed owned model section.
            native (object): Explicit original native model.
            frontend (str): Qualified original frontend.


        Raises:
            AssertionError: If original native assignment slots were modified.
        """
        targets = model.native_mapping(native)
        if frontend == "pyomo":
            assert all(target.value is None for target in targets.values())
        elif frontend == "pulp":
            assert all(target.varValue is None for target in targets.values())

    @staticmethod
    def _witness(
        model: ModelResult,
        result: RKOResult,
        scoring: str,
        reference: dict[str, Any],
        witness_check: dict[str, Any] | None,
    ) -> None:
        """Validate completed witness timing and scoring consistency.

        Args:
            model (ModelResult): Completed model evidence.
            result (RKOResult): Actual scalar search result.
            scoring (str): Selected score contract.
            reference (dict[str, Any]): Independent original oracle.
            witness_check (dict[str, Any] | None): Independent original witness check.


        Raises:
            AssertionError: If completed model evidence is inconsistent.
        """
        witness = model.best_feasible
        if witness is not None:
            assert witness_check["feasible"]
            assert 0 < model.time_to_first_feasible_seconds <= result.elapsed_seconds
            if scoring == "feasibility_first":
                assert model.search_incumbent.feasible
            assert reference["objective"] is not None
        else:
            assert model.time_to_first_feasible_seconds is None

    @staticmethod
    def _gap(
        case: PracticalCase, witness: ModelSolution | None, reference: dict[str, Any]
    ) -> float | None:
        """Compute the original-direction signed relative reference gap.

        Args:
            case (PracticalCase): Original mathematical formulation.
            witness (ModelSolution | None): Completed feasible original witness.
            reference (dict[str, Any]): Independent original optimum.

        Returns:
            float | None: Validated original campaign data.
        """
        gap = None
        if witness is not None:
            difference = (
                reference["objective"] - witness.objective
                if case.maximize
                else witness.objective - reference["objective"]
            )
            gap = 100 * difference / max(1.0, abs(reference["objective"]))
        return gap

    @staticmethod
    def _search_settings(seeds: int, time_limit: float) -> None:
        """Validate preplanned search settings.

        Args:
            seeds (int): Positive preplanned seed count.
            time_limit (float): Positive finite per-search duration.


        Raises:
            ValueError: If seed count or search duration is invalid.
        """
        if (
            type(seeds) is not int
            or seeds < 1
            or isinstance(time_limit, bool)
            or (not isinstance(time_limit, Real))
            or (not math.isfinite(time_limit))
            or (time_limit <= 0)
        ):
            raise ValueError("Positive seeds and a finite positive time limit are required.")

    @staticmethod
    def _repair_settings(repair: str, repair_max_sweeps: int) -> None:
        """Validate common restoration settings without per-case tuning.

        Args:
            repair (str): Common restoration policy.
            repair_max_sweeps (int): Common deterministic work cap.


        Raises:
            ValueError: If restoration policy or sweep cap is invalid.
        """
        if (
            repair not in ("none", "continuous")
            or type(repair_max_sweeps) is not int
            or (not 1 <= repair_max_sweeps <= 10000)
        ):
            raise ValueError("Invalid common repair policy or sweep cap.")


if __name__ == "__main__":
    ModelValidationCampaign.main()
