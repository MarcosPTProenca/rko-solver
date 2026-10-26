"""Trace current IPR pool diversity and candidate work under a shared evaluation cap.

Objective:
- Separate usable-pair formation, redundant path work and initialization diversity.

Technical approach:
- Run production IPR with owned RNGs and normal SearchSession callback isolation.
- Charge optional ten-vector initialization and all ordinary seeds to the same cap.
- Observe production pool admission, with a separately labeled exact-vector control.

Flow (pseudocode):
1. Optionally evaluate and insert a larger random starting population.
2. Run actual IPR, stopping before the first decoder call beyond the evaluation cap.
3. Report distinct pool vectors, equal-parent pairs, endpoints and novel candidates.

Important criteria:
- No wall/CPU timing, feasibility claims, native dependency or production policy change.
- Larger-population mode still pays for IPR's unconditional extra initial evaluation.
- Duplicate rejection is an isolated single-thread experiment, NOT a production policy.
- A bounded outer-loop guard fails if pair formation regresses instead of waiting forever.
- Synthetic threshold scores are not Knapsack quality or full C++ parity measurements.

Inputs/Outputs:
- Input: sphere, threshold or flat workload; seed, dimension, evaluation cap and prefill.
- Output: JSON work/diversity/objective records with deterministic candidate fingerprints.
"""

import argparse
import hashlib
import inspect
import json
import math
import numpy as np
import platform
import subprocess
from pathlib import Path
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.ipr import IPR
from rko_solver.orchestration.search import SearchSession
from rko_solver.orchestration.search import _BudgetExpired
from typing import Any
from typing import override


class DistinctVectorPool(SolutionPool):
    """Reject exact vector duplicates in an isolated single-thread diagnostic control.

    Responsibilities:
    - Isolate duplicate admission from initial population size.
    - Preserve different vectors even when their objective values tie.

    Attributes:
        capacity (int): Unchanged ten-entry capacity inherited from SolutionPool.
    """

    @override
    def insert(self, entry: tuple[float, list[float]], source: str = "") -> None:
        """Skip an already stored vector without changing production pool behavior.

        Args:
            entry (tuple[float, list[float]]): Already evaluated finite cost and keys.
            source (str): Original producer label.

        Raises:
            ValueError: If the inherited pool rejects an invalid novel entry.
        """
        if any(keys == entry[1] for _, keys in self.get_elite(self.capacity)):
            return
        super().insert(entry, source)


class IPRProbe(IPR):
    """Observe production IPR without altering its pool or candidate transitions.

    Responsibilities:
    - Enforce a shared callback cap, including external initialization.
    - Count selected-pair and candidate diversity with detached byte fingerprints.

    Attributes:
        limit (int): Positive completed-evaluation cap.
        prefill (int): Zero for ordinary bootstrap, or at least two external seeds.
        session (SearchSession): Normal callback isolation and best-result recovery.
        in_path (bool): Distinguish seed evaluations from relinking work.
        parents (tuple[bytes, bytes]): Current pair of vector fingerprints.
    """

    def __init__(
        self,
        workload: str,
        seed: int,
        dimension: int,
        limit: int,
        prefill: int,
        *,
        distinct_pool: bool = False,
    ) -> None:
        """Validate the bounded experiment and retain independent observation state.

        Args:
            workload (str): Sphere, threshold or flat numerical objective.
            seed (int): Owned Python/NumPy RNG seed.
            dimension (int): Positive key count.
            limit (int): At least two total objective evaluations.
            prefill (int): Zero or a starting population between two and ten.
            distinct_pool (bool): Use an explicit diagnostic-only duplicate-admission control.

        Raises:
            ValueError: If the experiment or population bounds are invalid.
        """
        if workload not in ("sphere", "threshold", "flat"):
            raise ValueError("unknown workload")
        if dimension < 1 or limit < 2 or prefill not in (0, *range(2, 11)):
            raise ValueError("require positive dimension, limit>=2 and prefill=0 or 2..10")
        self.workload, self.seed = workload, seed
        self.limit, self.prefill = limit, prefill
        self.distinct_pool = distinct_pool
        self.in_path = False
        self.parents = (b"", b"")
        self.pairs = self.equal_pairs = self.outer_checks = 0
        self.endpoint_calls = self.intermediate_calls = 0
        self.seed_vectors: list[np.ndarray] = []
        self.seed_alleles: list[set[float]] = [set() for _ in range(dimension)]
        self.outside_seed_alleles = 0
        self.path_vectors: set[bytes] = set()
        self.decoded_vectors: set[bytes] = set()
        self.intermediates: set[bytes] = set()
        self.trace_hash = hashlib.sha256()
        self.session = SearchSession(self.observe, self.objective)
        pool = DistinctVectorPool(10) if distinct_pool else SolutionPool(10)
        super().__init__(pool, self.observe, self.objective, dimension)

    def observe(self, keys: np.ndarray) -> np.ndarray:
        """Stop before over-budget decoding and classify only completed candidate work.

        Args:
            keys (np.ndarray): Detached candidate supplied by SearchSession.

        Returns:
            np.ndarray: Boolean threshold payload or unchanged continuous candidate.

        Raises:
            _BudgetExpired: Once the shared evaluation cap has been completed.
        """
        if self.session.n_evaluations >= self.limit:
            raise _BudgetExpired
        vector = keys.tobytes()
        self.trace_hash.update(vector)
        self.decoded_vectors.add(
            (keys >= 0.5).tobytes() if self.workload == "threshold" else vector
        )
        if self.in_path:
            self.outside_seed_alleles += any(
                float(key) not in allowed
                for key, allowed in zip(keys, self.seed_alleles, strict=True)
            )
            self.path_vectors.add(vector)
            if vector in self.parents:
                self.endpoint_calls += 1
            else:
                self.intermediate_calls += 1
                self.intermediates.add(vector)
        else:
            self.seed_vectors.append(keys.copy())
            for key, allowed in zip(keys, self.seed_alleles, strict=True):
                allowed.add(float(key))
        return keys >= 0.5 if self.workload == "threshold" else keys

    def objective(self, keys: np.ndarray) -> float:
        """Evaluate one domain-neutral continuous, discrete or tied landscape.

        Args:
            keys (np.ndarray): Observed chromosome.

        Returns:
            float: Finite minimization score.
        """
        if self.workload == "sphere":
            return sum(float(key) * float(key) for key in keys)
        if self.workload == "threshold":
            return float(np.count_nonzero(keys >= 0.5))
        return 0.0

    @override
    def time_limit_reached(self, limit_time: float) -> bool:
        """Bound outer-loop checks separately if a regressed search stops evaluating.

        Args:
            limit_time (float): Infinite wall allowance; this is a fixed-work probe.

        Returns:
            bool: False until ordinary callbacks enforce the evaluation cap.

        Raises:
            AssertionError: If pair formation or objective progress stalls.
        """
        self.outer_checks += 1
        if self.outer_checks > self.limit + 2 or (
            self.outer_checks > 2 and self.solution_pool.size < 2
        ):
            raise AssertionError("IPR stalled before the evaluation cap")
        return super().time_limit_reached(limit_time)

    @override
    def _relink(self, atual_keys: np.ndarray, guia_keys: np.ndarray) -> tuple[np.ndarray, float]:
        """Observe chosen parent identities around the unchanged production relinker.

        Args:
            atual_keys (np.ndarray): Starting elite.
            guia_keys (np.ndarray): Guide elite.

        Returns:
            tuple[np.ndarray, float]: Unmodified production path result.

        Raises:
            Exception: Propagates objective stops and original callback failures.
        """
        self.parents = atual_keys.tobytes(), guia_keys.tobytes()
        self.pairs += 1
        self.equal_pairs += self.parents[0] == self.parents[1]
        self.in_path = True
        try:
            return super()._relink(atual_keys, guia_keys)
        finally:
            self.in_path = False

    @override
    def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
        """Charge optional larger-population seeds before normal production solve.

        Args:
            limit_time (float): Infinite wall allowance from the diagnostic runner.

        Returns:
            tuple[list[object], list[float], float]: Original IPR result.

        Raises:
            Exception: Propagates callback stopping or failure during either phase.
        """
        for _ in range(self.prefill):
            keys = RandomStreams.numpy_rng().random(self.tam_solution)
            cost = self.cost(self.decoder(keys))
            self.solution_pool.insert((cost, keys.tolist()), "prefill")
        return super().solve(limit_time)

    def run(self) -> dict[str, Any]:
        """Execute exactly the capped work and return numerical/diversity evidence.

        Returns:
            dict[str, Any]: Completed evaluations, path identities and objective metrics.

        Raises:
            AssertionError: If work accounting or incumbent recovery is inconsistent.
        """
        with RandomStreams(self.seed).activate():
            result = self.session.solve(self, math.inf)
        assert self.session.n_evaluations == self.limit
        assert result[2] == self.objective(np.asarray(result[1]))
        elite = self.solution_pool.get_elite(10)
        unique = {np.asarray(keys).tobytes() for _, keys in elite}
        initial = np.stack(self.seed_vectors)
        return {
            "workload": self.workload,
            "seed": self.seed,
            "dimension": self.tam_solution,
            "prefill": self.prefill,
            "distinct_pool_control": self.distinct_pool,
            "path_outside_seed_alleles": self.outside_seed_alleles,
            "evaluations": self.session.n_evaluations,
            "seed_evaluations": len(self.seed_vectors),
            "path_evaluations": self.endpoint_calls + self.intermediate_calls,
            "pairs_started": self.pairs,
            "equal_parent_pairs": self.equal_pairs,
            "endpoint_evaluations": self.endpoint_calls,
            "intermediate_evaluations": self.intermediate_calls,
            "unique_intermediates": len(self.intermediates),
            "unique_path_vectors": len(self.path_vectors),
            "unique_decoded_vectors": len(self.decoded_vectors),
            "pool_size": len(elite),
            "unique_pool_vectors": len(unique),
            "unique_pool_decoded": len(
                {
                    (np.asarray(keys) >= 0.5).tobytes()
                    if self.workload == "threshold"
                    else np.asarray(keys).tobytes()
                    for _, keys in elite
                }
            ),
            "best_seed_cost": min(self.objective(keys) for keys in initial),
            "best_cost": result[2],
            "trace_sha256": self.trace_hash.hexdigest(),
            "seed_prefix_sha256": hashlib.sha256(initial[:2].tobytes()).hexdigest(),
        }


def single_block_path() -> dict[str, Any]:
    """Count redundant kernel work when distinct parents differ in just one block.

    Returns:
        dict[str, Any]: Actual callback and unique-candidate counts for a 20-gene pair.

    Raises:
        AssertionError: If the returned path score does not match its chromosome.
    """
    observed: list[bytes] = []

    def score(keys: np.ndarray) -> float:
        """Retain detached vector identities while evaluating a numerical norm.

        Args:
            keys (np.ndarray): Actual production path candidate.

        Returns:
            float: Squared Euclidean norm.
        """
        observed.append(keys.tobytes())
        return sum(float(key) * float(key) for key in keys)

    start = np.full(20, 0.8)
    guide = start.copy()
    guide[:2] = 0.2
    algorithm = IPR(SolutionPool(), lambda keys: keys, score, 20)
    keys, value = algorithm._relink(start, guide)
    count = len(observed)
    assert value == sum(float(key) * float(key) for key in keys)
    return {
        "case": "single_differing_block",
        "dimension": 20,
        "evaluations": count,
        "unique_vectors": len(set(observed[:count])),
        "start_evaluations": observed[:count].count(start.tobytes()),
        "guide_evaluations": observed[:count].count(guide.tobytes()),
        "returned_guide": np.array_equal(keys, guide),
    }


def main() -> None:
    """Emit the bounded diagnostic matrix without local timing or benchmark access.

    Raises:
        ValueError: If positive experiment bounds are invalid.
        AssertionError: If any production trace violates fixed-work accounting.
        subprocess.CalledProcessError: If run outside a readable Git repository.
        OSError: If Git or a source file is unavailable.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=int, default=1024)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    if args.seeds < 1:
        raise ValueError("seeds must be positive")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "source_revision": revision,
                "solver_source_sha256": hashlib.sha256(
                    Path(inspect.getfile(IPR)).read_bytes()
                ).hexdigest(),
                "pool_source_sha256": hashlib.sha256(
                    Path(inspect.getfile(SolutionPool)).read_bytes()
                ).hexdigest(),
                "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "timing": False,
            }
        ),
        flush=True,
    )
    print(json.dumps(single_block_path()), flush=True)
    for workload in ("sphere", "threshold", "flat"):
        for seed in range(200, 200 + args.seeds):
            rows = [
                IPRProbe(
                    workload, seed, args.dimension, args.evaluations, size, distinct_pool=distinct
                ).run()
                for size in (0, 10)
                for distinct in (False, True)
            ]
            assert len({row["seed_prefix_sha256"] for row in rows}) == 1
            for row in rows:
                assert row["seed_evaluations"] + row["path_evaluations"] == args.evaluations
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
