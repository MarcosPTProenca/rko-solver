"""Compare opt-in shared startup at equal completed objective work, without timing.

Objective:
- Charge startup and search to the same total evaluation cap in each paired trial.

Technical approach:
- Use real serial RKO and pure numerical/permutation objectives.
- Freeze policy elapsed time while retaining a real emergency stopping guard.
- Stop through the private cooperative signal after the exact completed-work cap.

Flow (pseudocode):
1. Run default and prefilled startup with the same search seed and callback budget.
2. Verify startup/search totals, first search vector and best observed objective.
3. Emit provenance, deterministic traces and paired outcome counts by workload.

Important criteria:
- This diagnostic uses an internal stop seam, not a public evaluation-limit API.
- One rejected objective attempt may decode a candidate; record it, never time it.
- Different seed populations/path counts are intentional, not equal trajectories.
- No concurrent, C++-parity, feasibility, throughput or default-on claim is made.

Inputs/Outputs:
- Input: bounded dimensions, capacities, evaluation counts and paired seeds.
- Output: JSONL provenance/trials/summaries without wall or CPU timing metrics.
"""

import argparse
import hashlib
import json
import math
import numpy as np
import platform
import statistics
import subprocess
from pathlib import Path
from rko_solver import RKO
from rko_solver.metaheuristics.base import MetaheuristicBase
from rko_solver.orchestration.search import _BudgetExpired
from typing import Any
from unittest.mock import patch


class Trial:
    """Own one fixed-work experiment across initialization and search callbacks.

    Responsibilities:
    - Track every completed observation and preserve deterministic trace evidence.
    - Reject additional objective work without reporting elapsed-time comparisons.

    Attributes:
        algorithms (tuple[str, ...]): Search implementations included in the pilot.
        workloads (tuple[str, ...]): Pure objective families without external instances.
    """

    algorithms = ("IPR", "GRASP", "MultiStart")
    workloads = ("sphere", "rugged", "ordering", "flat")

    def __init__(
        self,
        algorithm: str,
        workload: str,
        seed: int,
        dimension: int,
        evaluations: int,
        capacity: int,
        prefill: bool,
    ) -> None:
        """Validate work bounds and retain one serial trial's settings.

        Args:
            algorithm (str): Supported stock solver name.
            workload (str): Supported pure objective family.
            seed (int): Nonnegative search/initialization seed source.
            dimension (int): Positive chromosome length.
            evaluations (int): Total completed objective cap, including startup.
            capacity (int): At least two, so standalone IPR has usable parents.
            prefill (bool): Explicit startup policy, overriding TOML.

        Raises:
            ValueError: For invalid names or experiment bounds.
        """
        if algorithm not in self.algorithms or workload not in self.workloads:
            raise ValueError("unsupported algorithm or workload")
        if seed < 0 or dimension < 1 or capacity < 2 or evaluations <= 2 * capacity - 1:
            raise ValueError("require seed>=0, dimension>=1, capacity>=2 and evaluations>2P-1")
        self.algorithm = algorithm
        self.workload = workload
        self.seed = seed
        self.dimension = dimension
        self.limit = evaluations
        self.capacity = capacity
        self.prefill = prefill
        self.calls = 0
        self.decoder_calls = 0
        self.exhausted = False
        self.current_digest = b""
        self.digests: list[bytes] = []
        self.best = math.inf
        self.running_best: list[float] = []
        self.curve: list[tuple[int, float]] = []
        self.cuts = {1, capacity, 2 * capacity - 1, evaluations // 2, evaluations}

    def decode(self, keys: np.ndarray) -> np.ndarray:
        """Count decoder work separately, including a final unscored attempt.

        Args:
            keys (np.ndarray): Session-isolated chromosome.

        Returns:
            np.ndarray: Continuous keys or stable decoded permutation.
        """
        self.decoder_calls += 1
        self.current_digest = hashlib.sha256(keys.tobytes()).digest()
        return np.argsort(keys, kind="stable") if self.workload == "ordering" else keys

    def objective(self, payload: np.ndarray) -> float:
        """Complete at most the requested observations across both runtime phases.

        Args:
            payload (np.ndarray): Pure decoder output.

        Returns:
            float: Finite numerical or ordering cost.

        Raises:
            _BudgetExpired: On the first objective attempt beyond the completed cap.
        """
        if self.calls == self.limit:
            self.exhausted = True
            raise _BudgetExpired
        if self.workload == "sphere":
            value = float(np.dot(payload, payload))
        elif self.workload == "rugged":
            values = 10 * payload - 5
            value = float(10 * self.dimension + np.sum(values**2 - 10 * np.cos(2 * np.pi * values)))
        elif self.workload == "ordering":
            value = float(np.dot(payload, np.arange(1, self.dimension + 1)))
        else:
            value = 0.0
        self.calls += 1
        self.digests.append(self.current_digest)
        self.best = min(self.best, value)
        self.running_best.append(self.best)
        if self.calls in self.cuts:
            self.curve.append((self.calls, self.best))
        return value

    def run(self) -> dict[str, Any]:
        """Execute production dispatch with a frozen policy clock and real stop guard.

        Returns:
            dict[str, Any]: Exact work accounting, objective and trace fingerprints.

        Raises:
            AssertionError: If a real-time guard fires or accounting/control checks fail.
        """
        real_elapsed = MetaheuristicBase.elapsed

        def guarded_stop(algorithm: MetaheuristicBase, duration: float) -> bool:
            """Keep the emergency wall guard without feeding time into search policy.

            Args:
                algorithm (MetaheuristicBase): Active production search.
                duration (float): Emergency limit, not experimental work budget.

            Returns:
                bool: Cancellation or actual guard exhaustion.
            """
            return (
                algorithm._cancelled is not None and algorithm._cancelled.is_set()
            ) or real_elapsed(algorithm) >= duration

        with (
            patch.object(MetaheuristicBase, "elapsed", return_value=0.0),
            patch.object(MetaheuristicBase, "time_limit_reached", guarded_stop),
        ):
            result = RKO(
                self.decode,
                self.objective,
                self.dimension,
                algorithms=[self.algorithm],
                control=0,
                execution="sequential",
                pool_size=self.capacity,
                prefill_pool=self.prefill,
                random_state=self.seed,
            ).solve(60.0)
        initial = result.initialization
        startup = initial.n_evaluations if initial is not None else 0
        search = sum(report.n_evaluations for report in result.searches)
        assert self.exhausted and self.calls == result.n_evaluations == self.limit
        assert startup + search == self.limit and result.cost == self.best
        assert self.decoder_calls == self.limit + 1
        if initial is not None:
            assert initial.stop_reason == "completed" and startup <= 2 * self.capacity - 1
        optimum = self.dimension * (self.dimension**2 - 1) / 6 if self.workload == "ordering" else 0
        return {
            "kind": "trial",
            "algorithm": self.algorithm,
            "workload": self.workload,
            "seed": self.seed,
            "dimension": self.dimension,
            "capacity": self.capacity,
            "prefill": self.prefill,
            "evaluations": self.calls,
            "startup_evaluations": startup,
            "search_evaluations": search,
            "startup_best_cost": self.running_best[startup - 1] if startup else None,
            "search_improved_startup": result.cost < self.running_best[startup - 1]
            if startup
            else None,
            "decoder_calls": self.decoder_calls,
            "unscored_decoder_calls": 1,
            "termination": "evaluation_budget",
            "cost": result.cost,
            "gap": result.cost - optimum,
            "best_curve": self.curve,
            "search_seed": result.searches[0].random_state,
            "first_search_sha256": self.digests[startup].hex(),
            "startup_seed_prefix_sha256": hashlib.sha256(
                b"".join(self.digests[: self.capacity])
            ).hexdigest()
            if initial is not None
            else None,
            "trace_sha256": hashlib.sha256(b"".join(self.digests)).hexdigest(),
            "best_keys_sha256": hashlib.sha256(np.asarray(result.keys).tobytes()).hexdigest(),
        }


def main() -> None:
    """Emit a paired pilot with bounded serial work and source-identified provenance.

    Raises:
        ValueError: If trial bounds or seed count are invalid.
        AssertionError: If seed controls or completed-work accounting differ.
        OSError: If source/provenance cannot be read.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--capacity", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    if args.seeds < 1:
        raise ValueError("seeds must be positive")
    root = Path(__file__).resolve().parents[1]
    sources = [
        "solver.py",
        "components/solution_pool.py",
        "components/random_streams.py",
        "orchestration/initialization.py",
        "orchestration/search.py",
        "metaheuristics/ipr.py",
        "metaheuristics/grasp.py",
        "metaheuristics/multi_start.py",
        "metaheuristics/base.py",
        "parameters/config.toml",
        "parameters/grasp.toml",
        "parameters/ipr.toml",
        "parameters/multi_start.toml",
    ]
    sources.extend(
        str(path.relative_to(root / "src/rko_solver"))
        for path in sorted((root / "src/rko_solver/components/local_search").glob("*.py"))
    )
    print(
        json.dumps(
            {
                "kind": "provenance",
                "source_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    text=True,
                    cwd=root,
                ).strip(),
                "source_sha256": {
                    path: hashlib.sha256((root / "src/rko_solver" / path).read_bytes()).hexdigest()
                    for path in sources
                },
                "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "timing": False,
                "backend": "serial",
                "control": 0,
                "policy_clock": "frozen_zero",
                "emergency_guard_seconds": 60,
                "budget": "completed startup plus search objectives",
            }
        ),
        flush=True,
    )
    startup_prefixes: dict[int, str] = {}
    for algorithm in Trial.algorithms:
        for workload in Trial.workloads:
            outcomes = {"better": 0, "tied": 0, "worse": 0}
            costs: list[list[float]] = [[], []]
            for seed in range(200, 200 + args.seeds):
                pair = [
                    Trial(
                        algorithm,
                        workload,
                        seed,
                        args.dimension,
                        args.evaluations,
                        args.capacity,
                        prefill,
                    ).run()
                    for prefill in (False, True)
                ]
                assert pair[0]["search_seed"] == pair[1]["search_seed"]
                assert pair[0]["first_search_sha256"] == pair[1]["first_search_sha256"]
                prefix = pair[1]["startup_seed_prefix_sha256"]
                assert startup_prefixes.setdefault(seed, prefix) == prefix
                for index, row in enumerate(pair):
                    costs[index].append(row["cost"])
                    print(json.dumps(row), flush=True)
                before, after = pair[0]["cost"], pair[1]["cost"]
                outcomes["better" if after < before else "worse" if after > before else "tied"] += 1
            print(
                json.dumps(
                    {
                        "kind": "summary",
                        "algorithm": algorithm,
                        "workload": workload,
                        "pairs": args.seeds,
                        **outcomes,
                        "default_median": statistics.median(costs[0]),
                        "prefilled_median": statistics.median(costs[1]),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
