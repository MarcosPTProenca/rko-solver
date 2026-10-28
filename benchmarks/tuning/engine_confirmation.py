"""Sealed replay of one fixed preset on the Python and C++ engines.

Objective:
- Validate the already selected cross-family preset on the two unmeasured engines without
  selecting another candidate, changing regression limits or recomputing calibration scales.

Technical approach:
- Verify the original protocol and training hashes before consuming its locked selection.
- Replay exactly the archived confirmation instances, checking file hashes and copying the
  original calibration scales into an engine-isolated work directory.
- Reuse the spawned scoring executor and paired confirmation gates from the original study.

Flow (pseudocode):
1. Validate the archived lock and freeze engine, source revision and reference fingerprint.
2. Recreate the confirmation files and verify every instance against its archived hash.
3. Score the locked candidate and archived default, then apply the unchanged gates per engine.

Important criteria:
- All engines must pass their own original gates before proposing shared default changes.
- Search evaluation limits exclude native pool initialization, which C++ always performs.
- The C++ reference also uses process-global unseeded rand(); seeds do not fully control it.
- Failed gates never trigger reselection, relaxed limits or changes to packaged parameters.

Inputs/Outputs:
- Input: the sealed reference, engine name, work directory and process count.
- Output: immutable protocol, replayed instances, resumable scores and per-engine decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from benchmarks.tuning.instances import InstanceBank
from benchmarks.tuning.preset import PresetStudy
from benchmarks.tuning.preset import PresetTask
from benchmarks.tuning.preset_analysis import PresetAnalysis
from benchmarks.tuning.preset_analysis import PresetDesign
from dataclasses import asdict
from pathlib import Path
from rko_solver import _native
from typing import Any


class EngineConfirmation:
    """Replay a locked candidate against its archived default without another selection step.

    Responsibilities:
    - Bind each engine's measurements to the sealed candidate, original instances and scales.
    - Reuse scoring and acceptance rules without importing test scores into candidate selection.

    Attributes:
        directory (Path): Root containing separate Python and C++ work directories.
        reference (Path): Immutable full measurement from the training-only selection study.
        data (dict[str, Any]): Parsed sealed reference.
        design (PresetDesign): Original scientific design and regression gates.
        fingerprint (str): Digest of the exact reference bytes.
    """

    def __init__(self, directory: Path, reference: Path) -> None:
        """Load the reference and verify that its selected configuration remains hash-bound.

        Args:
            directory (Path): Parent work directory, outside the source checkout.
            reference (Path): Full sealed measurement in JSON format.

        Raises:
            OSError: If the reference cannot be read.
            ValueError: If its JSON or selection hashes are invalid.
            KeyError: If required reference fields are missing.
        """
        self.directory = directory.resolve()
        self.reference = reference.resolve()
        self.data: dict[str, Any] = json.loads(self.reference.read_text(encoding="utf-8"))
        self.design = PresetDesign(**self.data["protocol"]["design"])
        self.fingerprint = hashlib.sha256(self.reference.read_bytes()).hexdigest()
        protocol = self.data["protocol"]
        selection = self.data["selection"]
        training = "".join(PresetStudy._json(row) + "\n" for row in self.data["training"])
        valid = (
            hashlib.sha256(PresetStudy._json(protocol).encode()).hexdigest()
            == selection["protocol_sha256"]
            and hashlib.sha256(training.encode()).hexdigest() == selection["training_sha256"]
            and selection["config"] == protocol["candidates"][selection["candidate"]]
            and protocol["test_solver_seed_base"] == 55_000
        )
        if not valid:
            raise ValueError("The archived selection or its binding hashes have changed.")

    def _metadata(self, engine: str) -> dict[str, Any]:
        """Build scientific inputs independent of the current reporting revision.

        Args:
            engine (str): `python` or `cpp`.

        Returns:
            dict[str, Any]: Frozen replay inputs, excluding the execution source revision.

        Raises:
            ValueError: If the engine is unsupported or the input changed after loading.
        """
        if engine not in ("python", "cpp"):
            raise ValueError("Cross-engine confirmation supports only python and cpp.")
        if hashlib.sha256(self.reference.read_bytes()).hexdigest() != self.fingerprint:
            raise ValueError("The archived reference changed after loading.")
        return {
            "schema": 1,
            "engine": engine,
            "algorithm": "SA",
            "adaptation": "fixed",
            "reference_sha256": self.fingerprint,
            "design": asdict(self.design),
            "selection": self.data["selection"],
            "default": self.data["protocol"]["candidates"]["default"],
            "search_budget_excludes_pool_initialization": True,
            "fully_seeded_reference": engine != "cpp",
        }

    def prepare(self, engine: str) -> list[PresetTask]:
        """Freeze one replay and recreate only hash-verified confirmation instances.

        Args:
            engine (str): Engine whose isolated replay directory to prepare.

        Returns:
            list[PresetTask]: Exactly the archived default and locked candidate for every cell.

        Raises:
            ValueError: If frozen inputs or an instance differ from the reference.
            OSError: If replay files cannot be written.
        """
        metadata = self._metadata(engine)
        metadata["source_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        root = self.directory / engine
        root.mkdir(parents=True, exist_ok=True)
        PresetStudy._seal(root / "protocol.json", metadata)
        tasks = []
        selected = self.data["selection"]
        configs = {"default": metadata["default"], selected["candidate"]: selected["config"]}
        for family, size in self.design.contexts:
            bank = InstanceBank(root, seed_offset=self.design.offset(family, size), size=size)
            scale = bank.directory / f"calibration-{family}.json"
            name = scale.relative_to(root).as_posix()
            PresetStudy._seal(scale, self.data["calibration"][name])
            for item in bank.generate(family, "test", self.design.test):
                relative = item.path.relative_to(root).as_posix()
                if (
                    hashlib.sha256(item.path.read_bytes()).hexdigest()
                    != self.data["instances"][relative]
                ):
                    raise ValueError(f"Replayed instance does not match the reference: {relative}")
            for evaluations in self.design.budgets:
                for candidate, config in configs.items():
                    tasks.append(
                        PresetTask(
                            str(root),
                            self.design,
                            "test",
                            family,
                            size,
                            evaluations,
                            candidate,
                            config,
                            engine,
                        )
                    )
        return tasks

    def run(self, engine: str, workers: int = 3) -> dict[str, Any]:
        """Resume real scoring without calibrating, training or choosing another candidate.

        Args:
            engine (str): Python or C++ solver.
            workers (int): Spawned process count.

        Returns:
            dict[str, Any]: Unchanged per-engine confirmation gates and paired bounds.

        Raises:
            ValueError: If worker count, frozen inputs or resumed records are invalid.
            RuntimeError: If C++ was requested but is not built.
            Exception: First scoring failure, with the remaining workers terminated.
        """
        if workers < 1:
            raise ValueError("At least one worker is required.")
        if engine == "cpp" and not _native.cpp_available():
            raise RuntimeError("The optional C++ engine is not built.")
        tasks = self.prepare(engine)
        runner = PresetStudy(self.directory / engine, self.reference, self.design)
        runner._execute(tasks, "test", workers)
        return self.report(engine)

    def report(self, engine: str) -> dict[str, Any]:
        """Recompute a complete engine's decision without reselection or a source rewrite.

        Args:
            engine (str): Engine whose completed records to inspect.

        Returns:
            dict[str, Any]: Paired gates, selection identity and reference fingerprint.

        Raises:
            ValueError: If protocol or records are incomplete or identify another engine.
            OSError: If the protocol or report cannot be read or written.
        """
        root = self.directory / engine
        protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
        inputs = {key: value for key, value in protocol.items() if key != "source_sha"}
        if inputs != json.loads(PresetStudy._json(self._metadata(engine))):
            raise ValueError("Frozen engine confirmation inputs differ.")
        records = PresetStudy._records(root / "test.jsonl")
        if any(row.get("engine") != engine for row in records):
            raise ValueError("Confirmation records identify another engine.")
        result = PresetAnalysis(self.design).confirmation(
            records, self.data["selection"]["candidate"]
        )
        result.update(
            engine=engine, selection=self.data["selection"], reference_sha256=self.fingerprint
        )
        (root / "summary.json").write_text(PresetStudy._json(result) + "\n", encoding="utf-8")
        return result


def main() -> None:
    """Run or report one immutable cross-engine confirmation from command-line arguments.

    Raises:
        ValueError: If frozen evidence or scoring results are invalid.
        SystemExit: If command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "general-sa-preset-reference.json",
    )
    parser.add_argument("--engine", choices=("python", "cpp"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    study = EngineConfirmation(args.directory, args.reference)
    result = (
        study.run(args.engine, args.workers) if args.command == "run" else study.report(args.engine)
    )
    print(PresetStudy._json(result))


if __name__ == "__main__":
    main()
