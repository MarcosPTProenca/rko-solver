"""Cold-process time to first evaluation for adaptive PSO initialization.

Objective:
- Separate whole-solve startup from the Q-state graph initialization phase.

Technical approach:
- Run guarded spawn solves with identical five-value-per-axis parameter files.
- Wrap only the worker's graph builder to measure original/indexed construction.
- Record the first decoder timestamp and independently verify its seeded keys.

Flow (pseudocode):
1. Create temporary valid configuration and register an instrumented worker.
2. Alternate original/indexed graph builders over three process seeds.
3. Validate results and first keys, then restore the worker entry point.

Important criteria:
- Both variants use the same checkout, environment and spawn architecture.
- Real timed trajectories and final costs need not match.
- A first callback may occur after its budget; this benchmark reports that overrun.
- Instrumentation and file logging are part of both variants' measured solve.

Inputs/Outputs:
- Input: synthetic 625-state online PSO configuration and a sphere objective.
- Output: JSONL whole-solve, graph and first-evaluation timings.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from q_learning import PairwiseController
from rko_solver import RKO
from rko_solver.components.adaptive_control.q_learning import QLearningController
from rko_solver.components.adaptive_control.q_learning import QState
from rko_solver.orchestration import processes

_RUN_WORKER = processes._worker


@dataclass
class FirstDecode:
    """Record one worker's first evaluation and measured graph phase.

    Responsibilities:
    - Persist first-candidate timing without changing its payload.

    Attributes:
        path (Path): Per-trial observation file.
        pairwise (bool): Use the original graph in this worker.
        graph_seconds (float): Worker-side controller graph construction time.
        seen (bool): Whether the first candidate was already recorded.
    """

    path: Path
    pairwise: bool
    graph_seconds: float = 0.0
    seen: bool = False

    def __call__(self, keys: np.ndarray) -> np.ndarray:
        """Record the first keys and return the unchanged numerical payload.

        Args:
            keys (np.ndarray): Candidate vector supplied by the search session.

        Returns:
            np.ndarray: Same candidate for the sphere objective.

        Raises:
            OSError: If the observation cannot be written.
        """
        if not self.seen:
            self.seen = True
            self.path.write_text(
                json.dumps(
                    {
                        "at": time.perf_counter(),
                        "graph_seconds": self.graph_seconds,
                        "keys_sha256": hashlib.sha256(keys.astype("<f8").tobytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
        return keys


def objective(keys: np.ndarray) -> float:
    """Evaluate an independently checkable finite sphere objective.

    Args:
        keys (np.ndarray): Decoded candidate vector.

    Returns:
        float: Squared Euclidean norm.
    """
    return float(np.dot(keys, keys))


def _timed_worker(*args: object) -> None:
    """Instrument graph creation inside an otherwise unchanged spawn worker.

    Args:
        args (object): Existing process-worker positional arguments.

    Raises:
        AssertionError: If the worker receives an unexpected decoder type.
    """
    decoder = args[3]
    assert isinstance(decoder, FirstDecode)
    original = QLearningController._create_states
    method = PairwiseController._create_states if decoder.pairwise else original

    def create(
        parameters: list[list[float]],
        state_filter: Callable[[list[float]], bool] | None = None,
    ) -> list[QState]:
        """Measure only state/edge construction without changing its arguments.

        Args:
            parameters (list[list[float]]): Online parameter axes.
            state_filter (Callable | None): Production joint-feasibility filter.

        Returns:
            list[QState]: Original or indexed controller graph.
        """
        started = time.perf_counter()
        states = method(parameters, state_filter)
        decoder.graph_seconds = time.perf_counter() - started
        return states

    QLearningController._create_states = staticmethod(create)
    try:
        _RUN_WORKER(*args)
    finally:
        QLearningController._create_states = staticmethod(original)


def main() -> None:
    """Measure six alternating real process solves against the same configuration.

    Raises:
        AssertionError: If a result or seeded first-candidate contract fails.
        OSError: If configuration or observation files cannot be accessed.
        RuntimeError: If a worker fails or returns no result.
    """
    original = processes._worker
    processes._worker = _timed_worker
    try:
        with tempfile.TemporaryDirectory(prefix="rko-startup-") as temporary:
            directory = Path(temporary)
            (directory / "config.toml").write_text(
                '[run]\ncontrol = 1\npool_size = 10\n[active_metaheuristics]\nlist = ["PSO"]\n',
                encoding="utf-8",
            )
            (directory / "pso.toml").write_text(
                "[offline]\np_size = 5\nc1 = 0.5\nc2 = 0.5\nw = 0.5\n"
                "[online]\np_size = [1,2,3,4,5]\nc1 = [0.1,0.3,0.5,0.7,0.9]\n"
                "c2 = [0.1,0.3,0.5,0.7,0.9]\nw = [0.1,0.2,0.3,0.4,0.5]\n",
                encoding="utf-8",
            )
            for repeat, seed in enumerate((42, 43, 44)):
                worker_seed = int(np.random.SeedSequence(seed).spawn(1)[0].generate_state(1)[0])
                expected = np.random.RandomState(worker_seed).random_sample(32)
                expected_hash = hashlib.sha256(expected.astype("<f8").tobytes()).hexdigest()
                for label in (
                    ("pairwise", "indexed") if repeat % 2 == 0 else ("indexed", "pairwise")
                ):
                    path = directory / f"first-{seed}-{label}.json"
                    started = time.perf_counter()
                    result = RKO(
                        FirstDecode(path, label == "pairwise"),
                        objective,
                        32,
                        params_dir=directory,
                        random_state=seed,
                    ).solve(0.1)
                    first = json.loads(path.read_text(encoding="utf-8"))
                    assert first["keys_sha256"] == expected_hash
                    keys = np.asarray(result.keys)
                    assert np.all((keys >= 0.0) & (keys < 1.0))
                    assert np.isclose(result.cost, objective(keys))
                    print(
                        json.dumps(
                            {
                                "variant": label,
                                "seed": seed,
                                "states": 625,
                                "budget_seconds": 0.1,
                                "elapsed_seconds": result.elapsed_seconds,
                                "first_decode_seconds": first["at"] - started,
                                "graph_seconds": first["graph_seconds"],
                                "first_keys_sha256": first["keys_sha256"],
                            }
                        ),
                        flush=True,
                    )
    finally:
        processes._worker = original


if __name__ == "__main__":
    main()
