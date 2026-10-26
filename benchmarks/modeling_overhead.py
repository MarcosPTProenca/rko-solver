"""Controlled model-feature callback-boundary and bounded-evidence IPC diagnostics.

Objective:
- Quantify disabled-path regression and enabled archive/transport costs without changing callbacks.

Technical approach:
- Load the exact pre-feature SearchSession from Git into the same interpreter.
- Pair fixed-work callbacks with randomized order, controlled GC and optional CPU affinity.
- Compare generated callbacks with archive off/on while preserving callback objects and results.
- Measure pickle costs and spawned send/unpickle/ack transport separately from process startup.

Flow (pseudocode):
1. Identify original/candidate sources, runtime, callbacks and measurement settings.
2. Calibrate fixed counts once using the control, warm both paths and collect paired trials.
3. Report paired median ratios and descriptive bootstrap intervals without whole-engine claims.
4. Transport bounded two-witness worker payloads and report bytes/latencies independently.

Important criteria:
- Opaque handwritten callback bodies/contracts are identical across control and candidate.
- No native model object, candidate trajectory or objective replay is used for IPC payloads.
- A 5% disabled-boundary protection threshold is declared before timing; inconclusive is not pass.
- Synchronous transport includes an acknowledgement per message, not actual solver throughput.
- GC/affinity are restored and output files are created exclusively.

Inputs/Outputs:
- Input: Git baseline, fixed workloads, trial count and minimum per-sample duration.
- Output: source-identified paired observations, protection findings and serialization/IPC costs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import numpy as np
import os
import statistics
import subprocess
import sys
import tempfile
import time
from .modeling_boundary import ModelBoundaryProbe
from .modeling_probe_provenance import ModelProbeProvenance
from .modeling_restoration import ModelRestorationProbe
from .modeling_transport import ModelTransportProbe
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from rko_solver.examples.instance_pipeline import SchedulingInstance
from rko_solver.examples.instance_pipeline import SchedulingProblem
from rko_solver.examples.run_kp_cpp_instances import decode_knapsack
from rko_solver.examples.run_kp_cpp_instances import default_instances_dir
from rko_solver.examples.run_kp_cpp_instances import make_cost
from rko_solver.examples.run_kp_cpp_instances import read_kp_instance
from rko_solver.modeling.linear import CompiledLinearModel
from rko_solver.modeling.linear import LinearRow
from rko_solver.modeling.linear import LinearVariable
from rko_solver.orchestration.search import SearchSession
from rko_solver.orchestration.types import CostFn
from rko_solver.orchestration.types import DecoderFn
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundaryWorkload:
    """Retain identical callback objects and input coordinates for paired boundary trials.

    Responsibilities:
    - Keep callback contracts unchanged across control/candidate sessions.

    Attributes:
        name (str): Explicit problem/regime identifier.
        decoder (DecoderFn): Same callable used by both variants.
        cost (CostFn): Same callable used by both variants.
        keys (np.ndarray): Shared read-only input, never mutated by trials.
        expected (float): Independently obtained callback cost before timed work.
        generated (bool): Whether generated-model evidence is available.
    """

    name: str
    decoder: DecoderFn
    cost: CostFn
    keys: np.ndarray
    expected: float
    generated: bool = False


class ModelOverheadProbe:
    """Measure opt-in model bookkeeping without claiming general engine acceleration.

    Responsibilities:
    - Compare fixed identical work and validate costs/counts/incumbent isolation.
    - Separate enabled retention, pickle, transport and startup measurements.

    Attributes:
        threshold (float): Predeclared disabled-boundary candidate/control protection ratio.
    """

    threshold = 1.05

    @staticmethod
    def scalar_decode(keys: np.ndarray) -> float:
        """Return one opaque scalar for a deliberately cheap callback regime.

        Args:
            keys (np.ndarray): Detached candidate coordinates.

        Returns:
            float: First coordinate without generated-model semantics.
        """
        return float(keys[0])

    @staticmethod
    def scalar_cost(value: float) -> float:
        """Return the unchanged cheap finite payload as its objective.

        Args:
            value (float): Decoder payload.

        Returns:
            float: Ordinary minimization objective.
        """
        return value

    @staticmethod
    def native_decode(keys: np.ndarray) -> np.ndarray:
        """Return an opaque array for a native numerical callback regime.

        Args:
            keys (np.ndarray): Detached candidate coordinates.

        Returns:
            np.ndarray: Same decoder payload, unchanged between variants.
        """
        return keys

    @staticmethod
    def native_cost(values: np.ndarray) -> float:
        """Execute unchanged native numerical work to protect expensive/native callbacks.

        Args:
            values (np.ndarray): Opaque numerical payload.

        Returns:
            float: Finite nonlinear scalar, unrelated to model-conversion scoring.
        """
        return float(np.dot(values, values) + np.sin(values).sum())

    @staticmethod
    def model(dimension: int, rows: int, *, weighted: bool = False) -> CompiledLinearModel:
        """Create a controlled sparse snapshot without any optional frontend runtime.

        Args:
            dimension (int): Number of continuous bounded variables, at least eight.
            rows (int): Number of short sparse constraint rows.
            weighted (bool): Produce two distinct score/feasible witnesses for transport when true.

        Returns:
            CompiledLinearModel: Owned callback source, identical within each paired experiment.
        """
        width = min(8, dimension)
        return CompiledLinearModel(
            tuple(LinearVariable(f"v{i}", f"v{i}", 0, 1) for i in range(dimension)),
            (10.0,) * dimension,
            tuple(
                LinearRow(
                    f"r{r}",
                    tuple(((r * 3 + c) % dimension, 1.0) for c in range(width)),
                    lower=width if weighted else 0,
                    upper=float("inf") if weighted else width,
                )
                for r in range(rows)
            ),
            scoring="weighted_penalty" if weighted else "feasibility_first",
            penalty_weight=0.01 if weighted else None,
        )

    @classmethod
    def workloads(cls) -> tuple[BoundaryWorkload, ...]:
        """Bind unrelated unchanged handwritten callbacks and two generated snapshot sizes.

        Returns:
            tuple[BoundaryWorkload, ...]: Cheap, scheduling, Knapsack, native and model regimes.

        Raises:
            OSError: If bundled Knapsack reference data cannot be read.
        """
        scheduling = SchedulingProblem(
            SchedulingInstance(
                "boundary-scheduling",
                tuple(i % 7 + 1 for i in range(32)),
                tuple(i * 3 + 5 for i in range(32)),
            )
        )
        instance = read_kp_instance(default_instances_dir() / "kp50.txt")
        bindings = [
            ("scalar_8", cls.scalar_decode, cls.scalar_cost, 8, False),
            ("scheduling_32", scheduling.decode, scheduling.cost, 32, False),
            ("knapsack_50", decode_knapsack, make_cost(instance), 50, False),
            ("native_array_4096", cls.native_decode, cls.native_cost, 4096, False),
        ]
        for dimension, rows in ((8, 4), (128, 64)):
            model = cls.model(dimension, rows)
            bindings.append(
                (f"generated_{dimension}_{rows}", model.decode, model.cost, dimension, True)
            )
        result = []
        for name, decoder, cost, dimension, generated in bindings:
            keys = np.linspace(0.13, 0.87, dimension)
            expected = float(cost(decoder(keys.copy())))
            keys.setflags(write=False)
            result.append(BoundaryWorkload(name, decoder, cost, keys, expected, generated))
        return tuple(result)

    @staticmethod
    def baseline(revision: str) -> tuple[type[SearchSession], str]:
        """Load the exact original shared boundary under an isolated package module name.

        Args:
            revision (str): Git reference containing the pre-feature SearchSession.

        Returns:
            tuple: Baseline class and SHA256 of the actual loaded source.

        Raises:
            subprocess.CalledProcessError: If the baseline Git object is unavailable.
            RuntimeError: If the temporary baseline module cannot be loaded.
        """
        source = subprocess.check_output(
            ["git", "show", f"{revision}:src/rko_solver/orchestration/search.py"],
            text=True,
            encoding="utf-8",
        )
        with tempfile.TemporaryDirectory(prefix="rko-boundary-baseline-") as directory:
            path = Path(directory) / "baseline.py"
            path.write_text(source, encoding="utf-8")
            name = "rko_solver.orchestration._modeling_overhead_baseline"
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Could not load the exact original boundary.")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return (module.SearchSession, hashlib.sha256(source.encode("utf-8")).hexdigest())

    @staticmethod
    def sample(
        session_type: type[SearchSession],
        workload: BoundaryWorkload,
        count: int,
        *,
        archive: bool = True,
    ) -> float:
        """Time fixed completed work with unchanged callbacks and restored GC state.

        Args:
            session_type (type[SearchSession]): Original or current callback boundary.
            workload (BoundaryWorkload): Identical callback objects, expected cost and coordinates.
            count (int): Number of completed decoder/objective evaluations.
            archive (bool): Private benchmark-only switch for generated archive comparison;
                callback objects and bodies are never wrapped or replaced.

        Returns:
            float: Untraced elapsed seconds, excluding callback/session construction and validation.

        Raises:
            AssertionError: If counts, costs, witness eligibility or detached incumbent differ.
        """
        session = session_type(workload.decoder, workload.cost)
        if not archive and isinstance(session, SearchSession):
            session._model = None
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            started = time.perf_counter()
            for _ in range(count):
                session.evaluate(session.decode(workload.keys))
            elapsed = time.perf_counter() - started
        finally:
            if was_enabled:
                gc.enable()
        assert session.n_evaluations == count
        assert session.best is not None and session.best[1] == workload.expected
        assert np.array_equal(session.best[0], workload.keys)
        if isinstance(session, SearchSession):
            if workload.generated and archive:
                assert session.model_evidence is not None
            else:
                assert session.model_evidence is None
        return elapsed

    @staticmethod
    def paired_summary(
        pairs: list[dict[str, float]], *, protection: bool = False
    ) -> dict[str, Any]:
        """Describe paired ratios without treating noise or an overlapping interval as acceptance.

        Args:
            pairs (list[dict[str, float]]): All preplanned control/candidate observations.
            protection (bool): Apply the predeclared disabled-boundary five-percent threshold.

        Returns:
            dict[str, Any]: Median timings/ratio and descriptive paired bootstrap interval/findings.
        """
        ratios = np.array([pair["candidate_seconds"] / pair["control_seconds"] for pair in pairs])
        generator = np.random.default_rng(1729)
        bootstrap = np.median(
            ratios[generator.integers(0, len(ratios), (10000, len(ratios)))], axis=1
        )
        low, high = (float(value) for value in np.percentile(bootstrap, (2.5, 97.5)))
        finding = "measured_only"
        if protection:
            finding = (
                "within_boundary_threshold"
                if high <= ModelOverheadProbe.threshold
                else "regression"
                if low > ModelOverheadProbe.threshold
                else "inconclusive"
            )
        return {
            "control_median_seconds": statistics.median(p["control_seconds"] for p in pairs),
            "candidate_median_seconds": statistics.median(p["candidate_seconds"] for p in pairs),
            "paired_ratio_median": float(np.median(ratios)),
            "bootstrap95_ratio": [low, high],
            "finding": finding,
        }

    @staticmethod
    def echo(connection: Connection) -> None:
        """Unpickle each bounded parent message and acknowledge it without evaluating objectives.

        Args:
            connection (Connection): Spawned benchmark duplex pipe.

        Raises:
            EOFError: If the parent exits before orderly shutdown.
            ValueError: If an unexpected message type reaches the receiver.
        """
        try:
            connection.send("ready")
            while True:
                message = connection.recv()
                if message is None:
                    break
                if message[0] not in ("result", "model_evidence"):
                    raise ValueError("Unexpected benchmark transport message.")
                connection.send(True)
        finally:
            connection.close()

    @classmethod
    def messages(
        cls, dimension: int, rows: int, searches: int
    ) -> tuple[list[object], list[object]]:
        """Create actual completed distinct score/feasible witnesses in bounded worker payloads.

        Args:
            dimension (int): Number of variables/keys in each witness.
            rows (int): Number of constraint diagnostics per witness.
            searches (int): Number of worker sessions, not number of evaluated candidates.

        Returns:
            tuple: Legacy result messages and the same result plus independent evidence message.
        """
        model = cls.model(dimension, rows, weighted=True)
        evidence = []
        for _ in range(searches):
            session = SearchSession(model.decode, model.cost)
            for value in (0.0, 1.0):
                session.evaluate(session.decode(np.full(dimension, value)))
            record = session.model_evidence
            assert record is not None and record.best_feasible is not None
            assert not record.incumbent.feasible and record.best_feasible.feasible
            assert record.incumbent.score < record.best_feasible.score
            evidence.append(record)
        result = ([], list(evidence[0].incumbent.keys), evidence[0].incumbent.score)
        baseline = [("result", result)]
        return (baseline, [("model_evidence", evidence), *baseline])

    @staticmethod
    def transport_sample(connection: Connection, messages: list[object], count: int) -> float:
        """Time synchronous send/unpickle/ack rounds after process startup and warming.

        Args:
            connection (Connection): Ready child transport endpoint.
            messages (list[object]): Actual result/evidence message objects, without native models.
            count (int): Logical worker-result rounds, each containing all variant messages.

        Returns:
            float: Elapsed seconds for all logical rounds.

        Raises:
            TimeoutError: If the spawned receiver does not acknowledge a bounded message.
            AssertionError: If the receiver returns an invalid acknowledgement.
        """
        started = time.perf_counter()
        for _ in range(count):
            for message in messages:
                connection.send(message)
                if not connection.poll(30):
                    raise TimeoutError("Timed out waiting for bounded transport acknowledgement.")
                assert connection.recv() is True
        return time.perf_counter() - started

    @classmethod
    def ipc(cls, repeats: int) -> list[dict[str, Any]]:
        """Measure serialization and spawned transport separately for bounded evidence sizes.

        Args:
            repeats (int): Preplanned paired trial count.

        Returns:
            list[dict[str, Any]]: Payload bytes, pickle costs, startup and per-round observations.

        Raises:
            TimeoutError: If process startup or transport does not complete.
            RuntimeError: If a spawned receiver exits unexpectedly.
        """
        return ModelTransportProbe(cls, repeats).run()

    @staticmethod
    def restoration() -> list[dict[str, Any]]:
        """Measure explicit restoration work/cost, not an identical-semantics engine speedup.

        Returns:
            list[dict[str, Any]]: Fixed-key policy timings, completed sweeps and independent checks.

        Raises:
            AssertionError: If an original-model objective or feasibility check disagrees.
            OSError: If bundled fixture data cannot be read.
        """
        return ModelRestorationProbe.run()

    @classmethod
    def run(
        cls,
        output: Path,
        *,
        revision: str = "8917000",
        repeats: int = 21,
        target_seconds: float = 0.12,
    ) -> dict[str, Any]:
        """Run controlled fixed-work and IPC components, restoring local affinity and GC state.

        Args:
            output (Path): Exclusive new JSON reference/diagnostic output.
            revision (str): Exact pre-feature source reference.
            repeats (int): Positive preplanned paired repetitions.
            target_seconds (float): Positive baseline calibration target per fixed-work sample.

        Returns:
            dict[str, Any]: Source-identified raw observations and descriptive protection findings.

        Raises:
            FileExistsError: If output already exists.
            ValueError: If sample settings are invalid.
            OSError: If source/input/output/affinity operations fail.
            RuntimeError: If baseline loading or validation fails.
        """
        ModelBoundaryProbe.settings(repeats, target_seconds)
        if output.exists():
            raise FileExistsError(output)
        baseline, baseline_hash = cls.baseline(revision)
        previous_affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
        affinity = None
        data = ModelProbeProvenance.create(
            revision, repeats, target_seconds, baseline_hash, cls.threshold
        )
        sampler = ModelBoundaryProbe(cls, baseline, repeats, target_seconds)
        try:
            if previous_affinity:
                affinity = min(previous_affinity)
                os.sched_setaffinity(0, {affinity})
            data["pinned_cpu"] = affinity
            data["original_affinity"] = sorted(previous_affinity) if previous_affinity else None
            for workload in cls.workloads():
                data["boundary"].append(sampler.measure(workload))
            data["restoration"] = cls.restoration()
            data["ipc"] = cls.ipc(repeats)
        finally:
            if previous_affinity:
                os.sched_setaffinity(0, previous_affinity)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return data

    @classmethod
    def main(cls) -> None:
        """Run a source-identified local component qualification without overwriting measurements.

        Raises:
            OSError: If source/input/output/process resources are unavailable.
            ValueError: If requested settings are invalid.
            RuntimeError: If baseline or generated behavior fails validation.
        """
        parser = argparse.ArgumentParser(description="Measure model archive/disabled/IPC overhead.")
        parser.add_argument(
            "--output", type=Path, default=Path("docs/audits/modeling-overhead.json")
        )
        parser.add_argument("--repeats", type=int, default=21)
        parser.add_argument("--target-seconds", type=float, default=0.12)
        parser.add_argument("--baseline", default="8917000")
        args = parser.parse_args()
        result = cls.run(
            args.output,
            revision=args.baseline,
            repeats=args.repeats,
            target_seconds=args.target_seconds,
        )
        print(
            json.dumps(
                {
                    "boundary": [
                        {"workload": row["workload"], **row["summary"]}
                        for row in result["boundary"]
                    ],
                    "output": str(args.output),
                },
                indent=2,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    ModelOverheadProbe.main()
