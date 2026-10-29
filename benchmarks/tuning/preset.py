"""Training-only selection and sealed confirmation of one general fixed-parameter preset.

Objective:
- Compare one configuration across families, sizes and budgets without turning the confirmation
  set into another tuning set or changing any production parameter file.

Technical approach:
- Read every previously trained simulated-annealing configuration without inspecting historical
  test scores, deduplicate configurations and include the packaged fixed default.
- Freeze the design and candidate pool, score all candidates on training instances, then write
  a hash-bound selection before generating any confirmation instance.
- Reuse the existing compiled-objective instance bank and run independent cells in spawned
  workers. Persist completed records so interrupted runs resume with identical inputs.

Flow (pseudocode):
1. Freeze protocol, source identity and candidate pool; generate training instances only.
2. Rank candidates by the frozen family-balanced rule and seal one selection.
3. Generate untouched test instances, score the locked candidate and default, and report gates.

Important criteria:
- Candidate selection reads only training records; confirmation never reselects a candidate.
- Instance files and calibration are isolated by context, size and role.
- Workers reuse compiled problems, and the first failure terminates the pool and queued work.

Inputs/Outputs:
- Input: a work directory, a committed source of configurations and a frozen design.
- Output: protocol, training records, a sealed selection, test records and a decision summary.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import multiprocessing
import numpy as np
import subprocess
import sys
import time
from benchmarks.tuning.instances import InstanceBank
from benchmarks.tuning.preset_analysis import PresetAnalysis
from benchmarks.tuning.preset_analysis import PresetDesign
from benchmarks.tuning.spaces import SPACES
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from rko_solver.components.validation import ParameterValidator
from typing import Any


@dataclass(frozen=True)
class PresetTask:
    """Describe one phase, candidate and family-size-budget scoring cell.

    Responsibilities:
    - Carry all scoring inputs explicitly across the spawned-worker boundary.

    Attributes:
        directory (str): Study work directory.
        design (PresetDesign): Frozen scientific settings.
        phase (str): `train` or `test`.
        family (str): Problem family.
        size (int): Number of problem variables.
        evaluations (int): Evaluation limit per solver run.
        candidate (str): Stable configuration identifier.
        config (dict[str, float]): Fixed simulated-annealing parameters.
        engine (str): Solver engine, defaulting to the original Rust study.
    """

    directory: str
    design: PresetDesign
    phase: str
    family: str
    size: int
    evaluations: int
    candidate: str
    config: dict[str, float]
    engine: str = "rust"


@functools.lru_cache(maxsize=2)
def _bank(directory: str, offset: int, size: int) -> InstanceBank:
    """Reuse at most two compiled-problem banks in one worker.

    Args:
        directory (str): Study work directory.
        offset (int): Context-specific seed offset.
        size (int): Explicit problem dimension.

    Returns:
        InstanceBank: Bank isolated by cohort and size.
    """
    return InstanceBank(Path(directory), seed_offset=offset, size=size)


def prepare_context(payload: tuple[str, PresetDesign, str, str, int]) -> str:
    """Generate and calibrate one context in exactly one data role.

    Args:
        payload (tuple): Directory, frozen design, phase, family and dimension.

    Returns:
        str: Prepared context label.
    """
    directory, design, phase, family, size = payload
    count = design.train if phase == "train" else design.test
    bank = _bank(directory, design.offset(family, size), size)
    bank.calibrate(bank.generate(family, phase, count))
    label = f"{phase}/{family}/{size}"
    print(f"prepared {label}", file=sys.stderr, flush=True)
    return label


def score_cell(task: PresetTask) -> dict[str, Any]:
    """Score one configuration with shared solver seeds and calibrated instance scales.

    Args:
        task (PresetTask): Complete cell specification.

    Returns:
        dict[str, Any]: Normalized gaps and raw mean costs, one value per instance.

    Raises:
        ValueError: If a solver configuration is invalid or a cost is nonfinite.
    """
    design = task.design
    label = f"{task.phase}/{task.family}/{task.size}/{task.evaluations}/{task.candidate}"
    print(f"{time.strftime('%H:%M:%S')} start {label}", file=sys.stderr, flush=True)
    bank = _bank(task.directory, design.offset(task.family, task.size), task.size)
    bank.evaluations = task.evaluations
    count = design.train if task.phase == "train" else design.test
    seeds = design.train_seeds if task.phase == "train" else design.test_seeds
    base_seed = 44_000 if task.phase == "train" else 55_000
    items = bank.generate(task.family, task.phase, count)
    costs = []
    for index, item in enumerate(items, 1):
        costs.append(
            float(
                np.mean(
                    [
                        bank.cost(item, "SA", task.config, base_seed + s, engine=task.engine)
                        for s in range(seeds)
                    ]
                )
            )
        )
        print(
            f"{time.strftime('%H:%M:%S')} progress {label} {index}/{count}",
            file=sys.stderr,
            flush=True,
        )
    scores = [item.normalize(cost) for item, cost in zip(items, costs, strict=True)]
    print(f"{time.strftime('%H:%M:%S')} end {label}", file=sys.stderr, flush=True)
    return {
        "kind": "preset-score",
        "engine": task.engine,
        "phase": task.phase,
        "family": task.family,
        "size": task.size,
        "evaluations": task.evaluations,
        "candidate": task.candidate,
        "scores": scores,
        "costs": costs,
    }


class PresetStudy:
    """Freeze, select and confirm one fixed preset while preserving phase boundaries.

    Responsibilities:
    - Persist immutable study inputs and a training-only selection before test generation.
    - Execute resumable scoring and report the predeclared confirmation gates.

    Attributes:
        directory (Path): Work directory, never the source checkout.
        source (Path): Existing result file supplying configurations, not test evidence.
        design (PresetDesign): Frozen experimental settings.
    """

    def __init__(self, directory: Path, source: Path, design: PresetDesign | None = None) -> None:
        """Store the study paths and frozen settings without generating instances.

        Args:
            directory (Path): Work directory.
            source (Path): Candidate configuration source.
            design (PresetDesign | None): Scientific settings, or the predefined full study.
        """
        self.directory = directory
        self.source = source
        self.design = design or PresetDesign()

    def pool(self) -> dict[str, dict[str, float]]:
        """Read all trained configurations without reading historical test-score fields.

        Returns:
            dict[str, dict[str, float]]: Configuration identifiers and values, including default.

        Raises:
            ValueError: If the source has no trained configurations or any configuration is invalid.
        """
        defaults = {k: float(v) for k, v in SPACES["SA"].default().items()}
        pool = {"default": defaults}
        for row in self._records(self.source):
            if row.get("algorithm") != "SA" or row.get("tuner") != "racing":
                continue
            config = {k: float(v) for k, v in row["config"].items()}
            if config.keys() != defaults.keys():
                raise ValueError("Candidate keys do not match the fixed parameter schema.")
            ParameterValidator._values("SA", config)
            if config != defaults:
                key = hashlib.sha256(self._json(config).encode()).hexdigest()[:16]
                pool[key] = config
        if len(pool) == 1:
            raise ValueError("The source contains no trained candidate configurations.")
        return dict(sorted(pool.items()))

    def freeze(self) -> dict[str, Any]:
        """Bind the design and pool to source identity, refusing incompatible resumption.

        Returns:
            dict[str, Any]: Frozen protocol metadata.

        Raises:
            ValueError: If an existing protocol differs from the requested inputs.
        """
        protocol = {
            "schema": 1,
            "design": asdict(self.design),
            "candidates": self.pool(),
            "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "candidate_source": self.source.name,
            "candidate_source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
            "algorithm": "SA",
            "adaptation": "fixed",
            "engine": "rust",
            "train_solver_seed_base": 44_000,
            "test_solver_seed_base": 55_000,
        }
        self._seal(self.directory / "protocol.json", protocol)
        return json.loads(self._json(protocol))

    def select(self) -> dict[str, Any]:
        """Seal the best eligible candidate using only complete training evidence.

        Returns:
            dict[str, Any]: Chosen configuration with protocol and training evidence hashes.

        Raises:
            ValueError: If evidence is incomplete or an existing selection no longer matches it.
        """
        protocol = self.freeze()
        path = self.directory / "train.jsonl"
        ranked = PresetAnalysis(self.design).ranking(
            self._records(path), set(protocol["candidates"])
        )
        winner = ranked[0]
        selection = {
            **winner,
            "config": protocol["candidates"][winner["candidate"]],
            "protocol_sha256": hashlib.sha256(self._json(protocol).encode()).hexdigest(),
            "training_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        self._seal(self.directory / "selection.json", selection)
        return selection

    def run(self, stage: str, workers: int = 3) -> None:
        """Run training selection or locked-candidate confirmation, never retuning on test data.

        Args:
            stage (str): `select` or `confirm`.
            workers (int): Number of independent spawned workers.

        Raises:
            ValueError: If the stage, settings, existing evidence or worker count is invalid.
            Exception: First worker failure, after terminating all active and queued work.
        """
        protocol = self.freeze()
        if stage not in ("select", "confirm") or workers < 1:
            raise ValueError("Unknown stage or invalid worker count.")
        if stage == "select":
            phase, candidates = "train", protocol["candidates"]
        else:
            selected = self.select()
            phase = "test"
            candidates = {
                "default": protocol["candidates"]["default"],
                selected["candidate"]: selected["config"],
            }
        self._prepare(phase, workers)
        tasks = [
            PresetTask(str(self.directory), self.design, phase, family, size, budget, name, config)
            for family, size, budget in self.design.cases
            for name, config in candidates.items()
        ]
        self._execute(tasks, phase, workers)
        if stage == "select":
            print(self._json(self.select()), flush=True)
        else:
            self.report()

    def _prepare(self, phase: str, workers: int) -> None:
        """Calibrate only the requested role, after its phase boundary has been checked.

        Args:
            phase (str): Training or confirmation role.
            workers (int): Worker-process count.
        """
        context = multiprocessing.get_context("spawn")
        payloads = [
            (str(self.directory), self.design, phase, family, size)
            for family, size in self.design.contexts
        ]
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            futures = [pool.submit(prepare_context, item) for item in payloads]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                pool.terminate_workers()
                raise

    def _execute(self, tasks: list[PresetTask], phase: str, workers: int) -> None:
        """Append each missing cell once and terminate the pool immediately on failure.

        Args:
            tasks (list[PresetTask]): Complete phase task list.
            phase (str): Training or confirmation role.
            workers (int): Worker-process count.

        Raises:
            ValueError: If existing records contain duplicates or unexpected cells.
            Exception: First worker failure, without silently processing the remaining queue.
        """
        path = self.directory / f"{phase}.jsonl"
        known = self._completed(path, tasks)
        todo = [
            task
            for task in tasks
            if (task.candidate, task.family, task.size, task.evaluations) not in known
        ]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            futures = [pool.submit(score_cell, task) for task in todo]
            try:
                for future in as_completed(futures):
                    record = future.result()
                    with path.open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(self._json(record) + "\n")
            except BaseException:
                pool.terminate_workers()
                raise

    def _completed(self, path: Path, tasks: list[PresetTask]) -> set[tuple[str, str, int, int]]:
        """Validate existing keys before resuming a phase.

        Args:
            path (Path): Phase score file.
            tasks (list[PresetTask]): Allowed tasks.

        Returns:
            set[tuple]: Completed candidate, family, size and evaluation-budget keys.

        Raises:
            ValueError: If records duplicate a key or do not belong to the phase.
        """
        expected = {(t.candidate, t.family, t.size, t.evaluations): t.engine for t in tasks}
        seen = set()
        for row in self._records(path):
            key = (row["candidate"], row["family"], row["size"], row["evaluations"])
            if (
                key in seen
                or key not in expected
                or row["phase"] != tasks[0].phase
                or row.get("engine", "rust") != expected[key]
            ):
                raise ValueError("Duplicate or incompatible phase evidence.")
            seen.add(key)
        return seen

    def report(self) -> dict[str, Any]:
        """Write the locked candidate's confirmation decision without changing selection.

        Returns:
            dict[str, Any]: Paired confirmation bounds and acceptance gates.

        Raises:
            ValueError: If training or confirmation evidence is incomplete or inconsistent.
        """
        selected = self._locked_selection()
        result = PresetAnalysis(self.design).confirmation(
            self._records(self.directory / "test.jsonl"), selected["candidate"]
        )
        result["selection"] = selected
        (self.directory / "summary.json").write_text(self._json(result) + "\n", encoding="utf-8")
        return result

    def _locked_selection(self) -> dict[str, Any]:
        """Verify sealed evidence for read-only reporting without rerunning candidate selection.

        Returns:
            dict[str, Any]: Selection bound to the recorded protocol and training file.

        Raises:
            ValueError: If the selection, design or evidence hashes are inconsistent.
        """
        protocol = json.loads((self.directory / "protocol.json").read_text(encoding="utf-8"))
        selection = json.loads((self.directory / "selection.json").read_text(encoding="utf-8"))
        expected = {
            "protocol_sha256": hashlib.sha256(self._json(protocol).encode()).hexdigest(),
            "training_sha256": hashlib.sha256(
                (self.directory / "train.jsonl").read_bytes()
            ).hexdigest(),
        }
        if any(selection[key] != value for key, value in expected.items()):
            raise ValueError("Selection evidence hashes do not match.")
        if self._json(protocol["design"]) != self._json(asdict(self.design)):
            raise ValueError("Report design differs from the locked protocol.")
        if selection["config"] != protocol["candidates"][selection["candidate"]]:
            raise ValueError("Selected configuration differs from the protocol.")
        return selection

    @staticmethod
    def _records(path: Path) -> list[dict[str, Any]]:
        """Read newline-delimited records, preserving parse failures instead of skipping them.

        Args:
            path (Path): Input file, or an absent score file before its first task.

        Returns:
            list[dict[str, Any]]: Parsed records, or an empty list when the file is absent.
        """
        return (
            [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if path.exists()
            else []
        )

    @staticmethod
    def _json(value: object) -> str:
        """Encode deterministic, finite scientific metadata.

        Args:
            value (object): JSON-compatible value.

        Returns:
            str: Canonical JSON text.

        Raises:
            ValueError: If the value contains a nonfinite number.
        """
        return json.dumps(value, sort_keys=True, allow_nan=False)

    @classmethod
    def _seal(cls, path: Path, value: dict[str, Any]) -> None:
        """Create immutable metadata or verify an existing byte-equivalent logical value.

        Args:
            path (Path): Metadata file.
            value (dict[str, Any]): Expected value.

        Raises:
            ValueError: If existing metadata differs from the proposed value.
        """
        text = cls._json(value)
        if path.exists():
            if cls._json(json.loads(path.read_text(encoding="utf-8"))) != text:
                raise ValueError(f"Frozen evidence does not match: {path.name}.")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(text + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Execute an explicit selection, confirmation or reporting stage.

    Args:
        argv (Sequence[str] | None): Command arguments, or process arguments.

    Returns:
        int: Zero on successful execution, including an honestly failed scientific gate.

    Raises:
        SystemExit: If command arguments are invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("select", "confirm", "report"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("benchmarks/references/tuning-wide-expanded-reference.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    study = PresetStudy(args.directory, args.source)
    if args.stage == "report":
        print(study._json(study.report()))
    else:
        study.run(args.stage, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
