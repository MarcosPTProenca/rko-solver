"""Run tuning sessions in parallel and score their results on held-out instances.

Objective:
- Execute every (algorithm, family, tuner, repetition) session under the same budget, then
  score the returned configuration, the packaged default and a random configuration on the same
  held-out instances, and append one JSON line per session.

Technical approach:
- A session tunes on the training instances through a metered objective, then scores the chosen
  configuration on every held-out instance with several seeds that no tuner saw. The default and
  a random configuration are scored by the identical code with zero tuning runs, so the question
  "is tuning worth it" is answered by the same measurement as "which tuner is best".
- Sessions run in spawned worker processes (the Numba objectives compile once per worker) and
  results are appended as they finish, so a stopped bake-off resumes where it ended. Sessions are
  ordered by family and workers are replaced every few sessions, which bounds the memory that
  compiled objectives would otherwise accumulate.

Flow (pseudocode):
1. Generate and calibrate the training and held-out instances of each family.
2. Build the list of sessions that are not yet in the results file.
3. Run them in a process pool; each returns a record that is appended and flushed.

Important criteria:
- Instances, session seeds and held-out seeds are deterministic functions of the settings.
- Records hold per-instance held-out scores, so the analysis can resample instances.
- Held-out instances are never passed to a tuner.

Inputs/Outputs:
- Input: `Settings`, algorithm and family names, tuner names and a repetition count.
- Output: a JSON-lines file with one metadata line and one record per session.
"""

from __future__ import annotations

import functools
import json
import multiprocessing
import numpy as np
import time
import zlib
from benchmarks.tuning.external import OptunaTPE
from benchmarks.tuning.external import RKOTuner
from benchmarks.tuning.instances import FAMILIES
from benchmarks.tuning.instances import Instance
from benchmarks.tuning.instances import InstanceBank
from benchmarks.tuning.objective import TuningObjective
from benchmarks.tuning.spaces import SPACES
from benchmarks.tuning.tuners import IteratedRacing
from benchmarks.tuning.tuners import RandomHalving
from benchmarks.tuning.tuners import Tuner
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

TUNERS: dict[str, type[Tuner]] = {
    "random_halving": RandomHalving,
    "racing": IteratedRacing,
    "optuna_tpe": OptunaTPE,
    "rko": RKOTuner,
}
BASELINES = ("default", "random")
RECYCLE_AFTER = 6


@dataclass(frozen=True)
class Settings:
    """Parameters of one bake-off.

    Responsibilities:
    - Fix the budgets and sizes so that every session is comparable.

    Attributes:
        directory (str): Folder for instances and calibration.
        budget (int): Solver runs a tuner may spend.
        train (int): Training instances per family.
        test (int): Held-out instances per family.
        test_seeds (int): Solver seeds per held-out instance.
        evaluations (int): Evaluation limit of every solver run.
        workers (int): Worker processes.
    """

    directory: str
    budget: int = 1000
    train: int = 12
    test: int = 20
    test_seeds: int = 5
    evaluations: int = 50_000
    workers: int = 4


def session_seed(algorithm: str, family: str, tuner: str, rep: int) -> int:
    """Derive a stable seed for a session.

    Args:
        algorithm (str): Tuned algorithm.
        family (str): Problem family.
        tuner (str): Tuner or baseline name.
        rep (int): Repetition index.

    Returns:
        int: Deterministic non-negative seed.
    """
    return zlib.crc32(f"{algorithm}|{family}|{tuner}|{rep}".encode()) & 0x7FFFFFFF


@functools.lru_cache(maxsize=1)
def _worker_bank(settings: Settings) -> tuple[InstanceBank, dict[tuple[str, str], list[Instance]]]:
    """Return the per-process bank and instance lists, building them once.

    Args:
        settings (Settings): Bake-off settings.

    Returns:
        tuple: The bank and a mapping `(family, role)` to calibrated instances.
    """
    bank = InstanceBank(Path(settings.directory), settings.evaluations)
    instances = {
        (family, role): bank.generate(family, role, count)
        for family in FAMILIES
        for role, count in (("train", settings.train), ("test", settings.test))
    }
    return bank, instances


def _prepare_family(payload: tuple[Settings, str]) -> str:
    """Generate and calibrate the instances of one family in a worker.

    Args:
        payload (tuple[Settings, str]): Settings and family name.

    Returns:
        str: The family name, once its calibration is stored.
    """
    settings, family = payload
    bank = InstanceBank(Path(settings.directory), settings.evaluations)
    for role, count in (("train", settings.train), ("test", settings.test)):
        bank.calibrate(bank.generate(family, role, count))
    return family


def run_session(payload: tuple[Settings, str, str, str, int]) -> dict[str, object]:
    """Tune (or take a baseline) and score the result on the held-out instances.

    Args:
        payload (tuple): Settings, algorithm, family, tuner or baseline name and repetition.

    Returns:
        dict[str, object]: Record with the configuration, runs used and held-out scores.

    Raises:
        KeyError: If the algorithm, family or tuner is unknown.
    """
    settings, algorithm, family, tuner, rep = payload
    bank, instances = _worker_bank(settings)
    train, test = instances[(family, "train")], instances[(family, "test")]
    space = SPACES[algorithm]
    seed = session_seed(algorithm, family, tuner, rep)
    started = time.perf_counter()
    runs = 0
    if tuner == "default":
        config = space.default()
    elif tuner == "random":
        config = space.decode(np.random.default_rng(seed).random(space.dimension))
    else:
        objective = TuningObjective(
            lambda cfg, index, run_seed: bank.score(train[index], algorithm, cfg, run_seed),
            len(train),
            settings.budget,
            base_seed=seed % 100_000,
        )
        config = TUNERS[tuner]().tune(objective, space, seed)
        runs = objective.runs_used
    tuned_seconds = time.perf_counter() - started
    scores = [
        float(
            np.mean(
                [bank.score(item, algorithm, config, 9000 + n) for n in range(settings.test_seeds)]
            )
        )
        for item in test
    ]
    return {
        "kind": "session",
        "algorithm": algorithm,
        "family": family,
        "tuner": tuner,
        "rep": rep,
        "config": config,
        "runs_used": runs,
        "tune_seconds": round(tuned_seconds, 2),
        "test": scores,
    }


class BakeOff:
    """Prepare instances and run the missing sessions.

    Responsibilities:
    - Calibrate instances, list unfinished sessions and run them in a process pool.
    - Append each finished record to the results file.

    Attributes:
        settings (Settings): Bake-off settings.
        results (Path): JSON-lines results file.
    """

    def __init__(self, settings: Settings, results: Path) -> None:
        """Create a bake-off.

        Args:
            settings (Settings): Bake-off settings.
            results (Path): JSON-lines file, created when missing.
        """
        self.settings = settings
        self.results = results

    def prepare(self) -> None:
        """Generate and calibrate every family's instances, one family per process."""
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            min(len(FAMILIES), self.settings.workers), mp_context=context
        ) as pool:
            list(pool.map(_prepare_family, [(self.settings, family) for family in FAMILIES]))

    def done(self) -> set[tuple[str, str, str, int]]:
        """Return the sessions already in the results file.

        Returns:
            set[tuple[str, str, str, int]]: `(algorithm, family, tuner, rep)` keys.
        """
        if not self.results.is_file():
            return set()
        lines = [json.loads(line) for line in self.results.read_text(encoding="utf-8").splitlines()]
        return {
            (r["algorithm"], r["family"], r["tuner"], r["rep"])
            for r in lines
            if r.get("kind") == "session"
        }

    def pending(
        self,
        algorithms: Sequence[str],
        families: Sequence[str],
        tuners: Sequence[str],
        reps: int,
    ) -> list[tuple[Settings, str, str, str, int]]:
        """List the sessions that are not yet in the results file.

        Args:
            algorithms (Sequence[str]): Algorithms to tune.
            families (Sequence[str]): Problem families.
            tuners (Sequence[str]): Tuner names; baselines are added automatically.
            reps (int): Repetitions per tuner and baseline.

        Returns:
            list[tuple]: Session payloads for `run_session`.
        """
        finished = self.done()
        return [
            (self.settings, a, f, t, r)
            for f in families
            for a in algorithms
            for t in (*BASELINES, *tuners)
            for r in range(reps)
            if (a, f, t, r) not in finished and not (t == "default" and r > 0)
        ]

    def run(
        self,
        algorithms: Sequence[str],
        families: Sequence[str],
        tuners: Sequence[str],
        reps: int,
    ) -> int:
        """Run every session that is not yet recorded.

        Args:
            algorithms (Sequence[str]): Algorithms to tune.
            families (Sequence[str]): Problem families.
            tuners (Sequence[str]): Tuner names; baselines are added automatically.
            reps (int): Repetitions per tuner and baseline.

        Returns:
            int: Number of sessions executed in this call.

        Raises:
            KeyError: If a name is unknown.
        """
        if not self.results.is_file():
            self.results.parent.mkdir(parents=True, exist_ok=True)
            meta = {"kind": "meta", "settings": asdict(self.settings), "created": time.time()}
            self.results.write_text(json.dumps(meta) + "\n", encoding="utf-8")
        todo = self.pending(algorithms, families, tuners, reps)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            self.settings.workers, mp_context=context, max_tasks_per_child=RECYCLE_AFTER
        ) as pool:
            futures = [pool.submit(run_session, item) for item in todo]
            for future in as_completed(futures):
                with self.results.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(future.result()) + "\n")
        return len(todo)
