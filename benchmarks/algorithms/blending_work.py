"""Paired whole-search experiments with a trusted uncommitted blending candidate.

Objective:
- Test a candidate without committing an unvalidated production optimization.

Technical approach:
- Replace only Nelder-Mead's shared blending binding inside existing fixed work.
- Alternate baseline/candidate order and reject any changed replay fingerprint.
- Include deterministic Python-heavy callbacks and separate allocation diagnostics.

Flow (pseudocode):
1. Validate candidate offspring, ownership, errors and RNG against production.
2. Warm each variant and run paired SA/GA/BRKGA searches at fixed work.
3. Restore bindings and random streams even if the candidate fails.

Important criteria:
- Candidate modules execute arbitrary TRUSTED Python; never load untrusted inputs.
- Search timing excludes import, construction and process startup.
- Allocation diagnostics are not process RSS or acceptance timing measurements.

Inputs/Outputs:
- Input: trusted module exporting blending and bounded workload parameters.
- Output: JSONL provenance, allocation observations and exact-replay search pairs.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import numpy as np
import pickle
import random
import rko_solver.components.local_search.nelder_mead_ls as nelder_mead
import sys
import tracemalloc
from benchmarks.runtime.fixed_work import _run
from benchmarks.runtime.performance_profile import PerformanceProbe
from collections.abc import Callable
from pathlib import Path
from rko_solver.components.blending.blending import blending
from rko_solver.metaheuristics.brkga import BRKGA
from rko_solver.metaheuristics.ga import GeneticAlgorithm
from rko_solver.metaheuristics.sa import SA
from rko_solver.orchestration.search import SearchSession
from rko_solver.orchestration.types import CostFn
from rko_solver.orchestration.types import DecoderFn
from typing import override

Operator = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


class PythonWorkSession(SearchSession):
    """Add deterministic CPU-bound Python work without extra objective invocations.

    Responsibilities:
    - Preserve the actual objective result and completed-evaluation accounting.
    - Model callback overhead rather than sleeping or repeating stateful callbacks.
    """

    @override
    def __init__(self, decoder: DecoderFn, cost: CostFn) -> None:
        """Wrap scoring with a bounded Python arithmetic workload.

        Args:
            decoder (DecoderFn): Original pure synthetic decoder.
            cost (CostFn): Original evaluation-counting objective, called once.
        """

        def objective(payload: np.ndarray) -> float:
            """Compute finite deterministic overhead before one original objective.

            Args:
                payload (np.ndarray): Synthetic chromosome or permutation.

            Returns:
                float: Unchanged synthetic objective value.

            Raises:
                ValueError: If the synthetic arithmetic unexpectedly becomes nonfinite.
            """
            checksum = 0.0
            for _ in range(8):
                for value in payload:
                    checksum += float(value) * float(value)
            if not math.isfinite(checksum):
                raise ValueError("Nonfinite synthetic callback overhead.")
            return cost(payload)

        super().__init__(decoder, objective)


class BlendingStudy:
    """Verify a trusted operator and compare unchanged complete serial searches.

    Responsibilities:
    - Preserve candidate semantics, caller state and production bindings.
    - Reject changed replay instead of reporting a misleading speedup.
    """

    @staticmethod
    def validate(candidate: Operator) -> None:
        """Require exact offspring bits, random state, ownership and validation errors.

        Args:
            candidate (Operator): Trusted proposed replacement.

        Raises:
            ValueError: If the candidate changes any checked operator contract.
        """
        saved, saved_numpy = random.getstate(), np.random.get_state()
        try:
            for dimension in (0, 1, 20, 32, 499, 500, 1000):
                for dtype in (np.float32, np.float64):
                    a = np.linspace(0, 0.99, dimension, dtype=dtype)
                    b = a[::-1].copy()
                    for factor in (1.0, -1.0):
                        random.seed(777)
                        expected = blending(a, b, factor)
                        expected_state = random.getstate()
                        expected_numpy = pickle.dumps(np.random.get_state())
                        before_a, before_b = a.copy(), b.copy()
                        random.seed(777)
                        actual = candidate(a, b, factor)
                        if (
                            actual.dtype != expected.dtype
                            or actual.shape != expected.shape
                            or actual.tobytes() != expected.tobytes()
                            or random.getstate() != expected_state
                            or pickle.dumps(np.random.get_state()) != expected_numpy
                            or not np.array_equal(a, before_a)
                            or not np.array_equal(b, before_b)
                            or np.shares_memory(actual, a)
                            or np.shares_memory(actual, b)
                        ):
                            raise ValueError("Changed operator replay, ownership or RNG.")
            for a, b in ((np.zeros(2), np.zeros(3)), (np.zeros((1, 1)), np.zeros((1, 1)))):
                try:
                    candidate(a, b, 1.0)
                except ValueError:
                    pass
                else:
                    raise ValueError("Candidate omitted parent validation.")
        finally:
            random.setstate(saved)
            np.random.set_state(saved_numpy)

    @classmethod
    def run(
        cls,
        candidate: Operator,
        dimension: int = 500,
        evaluations: int = 8192,
        seeds: int = 5,
        seed_start: int = 200,
    ) -> list[dict[str, object]]:
        """Collect unprofiled fixed-work pairs with unchanged searches and callbacks.

        Args:
            candidate (Operator): Trusted candidate callable.
            dimension (int): Positive chromosome length.
            evaluations (int): Positive completed search evaluation count.
            seeds (int): Positive number of paired seeds per regime/algorithm.
            seed_start (int): First NumPy-compatible seed.

        Returns:
            list[dict[str, object]]: Paired raw results with order and workload identity.

        Raises:
            ValueError: If inputs or candidate replay are invalid.
            Exception: Propagates candidate or search errors after restoring state.
        """
        if any(type(value) is not int or value < 1 for value in (dimension, evaluations, seeds)):
            raise ValueError("dimension, evaluations and seeds must be positive integers.")
        if type(seed_start) is not int or not 0 <= seed_start <= 2**32 - seeds:
            raise ValueError("Invalid paired seed range.")
        cls.validate(candidate)
        saved_python, saved_numpy = random.getstate(), np.random.get_state()
        original = nelder_mead._blend
        rows = []
        try:
            for algorithm in (SA, GeneticAlgorithm, BRKGA):
                for regime, session in (
                    ("native", SearchSession),
                    ("python_heavy", PythonWorkSession),
                ):
                    for operator in (blending, candidate):
                        nelder_mead._blend = operator
                        _run(algorithm, "sphere", 7, session, dimension, min(evaluations, 64))
                    for seed in range(seed_start, seed_start + seeds):
                        order = (
                            ("baseline", "candidate")
                            if seed % 2 == 0
                            else ("candidate", "baseline")
                        )
                        runs = {}
                        for label in order:
                            nelder_mead._blend = blending if label == "baseline" else candidate
                            runs[label] = _run(
                                algorithm, "sphere", seed, session, dimension, evaluations
                            )
                        expected = {
                            key: value
                            for key, value in runs["baseline"].items()
                            if key not in ("seconds", "cpu_seconds")
                        }
                        actual = {
                            key: value
                            for key, value in runs["candidate"].items()
                            if key not in ("seconds", "cpu_seconds")
                        }
                        if actual != expected:
                            raise ValueError(
                                f"Changed search replay: {algorithm.__name__}/{regime}/{seed}."
                            )
                        rows.append(
                            {
                                "kind": "pair",
                                "regime": regime,
                                "order": order,
                                "baseline": runs["baseline"],
                                "candidate": runs["candidate"],
                            }
                        )
        finally:
            nelder_mead._blend = original
            random.setstate(saved_python)
            np.random.set_state(saved_numpy)
        return rows


def main() -> None:
    """Execute a TRUSTED candidate source and emit paired experimental evidence.

    Raises:
        ValueError: If candidate loading or checked search replay fails.
        OSError: If source cannot be read or output cannot be written.
        Exception: Propagates trusted candidate module and operator failures.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="TRUSTED Python source only.")
    parser.add_argument("--dimension", type=int, default=500)
    parser.add_argument("--evaluations", type=int, default=8192)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=200)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("_rko_trusted_blending_candidate", args.candidate)
    if spec is None or spec.loader is None:
        raise ValueError("Candidate must be trusted loadable Python source.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    candidate = module.blending
    print(
        json.dumps(
            {
                "kind": "environment",
                "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "provenance": PerformanceProbe.provenance(),
            }
        ),
        flush=True,
    )
    rows = BlendingStudy.run(
        candidate, args.dimension, args.evaluations, args.seeds, args.seed_start
    )
    for row in rows:
        print(json.dumps(row, allow_nan=False), flush=True)
    a = np.linspace(0, 0.99, args.dimension)
    b = a[::-1].copy()
    for label, operator in (("baseline", blending), ("candidate", candidate)):
        random.seed(42)
        tracemalloc.start()
        try:
            operator(a, b, -1.0)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        print(
            json.dumps(
                {
                    "kind": "operator_allocation",
                    "variant": label,
                    "dimension": args.dimension,
                    "peak_tracemalloc_bytes": peak,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
