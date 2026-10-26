"""Container-only decoder evaluation and isolation qualification probes.

Objective:
- Run a seed/candidate against immutable inputs using unchanged RKO callbacks.
Technical approach:
- Import generated source only inside the externally limited worker.
- Recompute every evaluated output and retain a bounded best witness.
Flow (pseudocode):
1. Read one bounded job or trusted qualification probe.
2. Execute fixed RKO or matched-runtime controls.
3. Emit one bounded JSON record for independent parent verification.
Important criteria:
- Never invoke this module on untrusted source outside PodmanRunner.
- Python co-residency is not an adversarial evaluator-integrity guarantee.
Inputs/Outputs:
- Input: one JSON job on stdin.
- Output: one JSON evidence record on stdout.
"""

from __future__ import annotations

import contextlib
import io
import json
import numpy as np
import os
import resource
import subprocess
import sys
import time
from ...components.random_streams import RandomStreams
from .artifacts import DecoderArtifact
from .problems import Problem
from collections.abc import Iterable
from rko_solver import RKO
from types import MappingProxyType
from typing import Any
from typing import cast


class _InnerDeadline(Exception):
    """Stop at the next callback boundary while retaining completed evidence.

    Responsibilities:
    - Signal cooperative inner-budget exhaustion to the worker execution phase.

    Attributes:
        args (tuple): Standard exception arguments.
    """


def _pid_probe() -> dict[str, Any]:
    """Attempt bounded process creation and reap all successfully created children.

    Returns:
        dict[str, Any]: Observed process-limit enforcement and child count.

    Raises:
        OSError: If child cleanup fails.
    """
    children = []
    blocked = False
    try:
        for _ in range(64):
            try:
                children.append(
                    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
                )
            except OSError:
                blocked = True
                break
    finally:
        for child in children:
            child.kill()
            child.wait()
    return {"pid_limit_enforced": blocked, "spawned": len(children)}


def _probe(name: str) -> dict[str, Any]:
    """Exercise PID enforcement or intentionally exhaust a resource.

    Args:
        name (str): Trusted hang/output/memory/pids probe name.

    Returns:
        dict[str, Any]: PID evidence; other probes must not return normally.

    Raises:
        ValueError: If probe name is unknown.
        MemoryError: If allocation exhaustion stops the memory probe.
        OSError: If output or child cleanup fails.
    """
    if name == "hang":
        while True:
            pass
    if name == "output":
        while True:
            os.write(1, b"x" * 16384)
    if name == "memory":
        blocks = []
        while True:
            blocks.append(bytearray(16 * 1024 * 1024))
    if name == "pids":
        return _pid_probe()
    raise ValueError("Unknown isolation probe.")


def _immutable(value: object) -> object:
    """Detach mutable containers into read-only mappings/tuples.

    Args:
        value (object): JSON-compatible input.

    Returns:
        object: Recursively immutable callback data.
    """
    if isinstance(value, dict):
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_immutable(item) for item in value)
    return value


class _JobEvaluation:
    """Own callback state and fixed execution phases for a single isolated job.

    Responsibilities:
    - Detach decoder inputs and independently evaluate each produced solution.
    - Retain a bounded best witness and execute the requested fixed control.

    Attributes:
        problem (Problem): Validated instance evaluator.
        instance (object): Immutable callback input.
        best (dict[str, Any] | None): Best completed evidence.
        evaluations (int): Completed evaluation count.
        infeasible (int): Completed infeasible evaluation count.
        distinct (set[tuple[int, ...]]): At most 256 distinct outputs.
        deadline (float): Cooperative callback deadline.
        started (float): Start of charged inner execution.
        decode (Any): Generated decoder callable loaded inside isolation.
    """

    def __init__(self, job: dict[str, Any]) -> None:
        """Validate source and prepare per-job callback state before timing execution.

        Args:
            job (dict[str, Any]): Controller source and instance envelope.

        Raises:
            ValueError: If source or instance validation fails.
            Exception: If generated module initialization fails.
        """
        artifact = DecoderArtifact.create(job["family"], job["source"])
        self.problem = Problem(job["family"], job["data"])
        namespace: dict[str, Any] = {}
        exec(compile(artifact.source, "decoder.py", "exec"), namespace)
        self.decode = namespace["decode"]
        self.instance = _immutable(self.problem.data)
        self.best: dict[str, Any] | None = None
        self.evaluations = 0
        self.infeasible = 0
        self.distinct: set[tuple[int, ...]] = set()
        self.started = 0.0
        self.deadline = 0.0

    def decoder(self, keys: np.ndarray) -> object:
        """Call generated code with detached read-only keys and immutable instance data.

        Args:
            keys (np.ndarray): Solver-produced key vector.

        Returns:
            object: Untrusted decision payload.

        Raises:
            _InnerDeadline: If a completed witness exists and the budget expired.
            Exception: If the candidate decoder fails.
        """
        if self.best is not None and time.perf_counter() >= self.deadline:
            raise _InnerDeadline
        snapshot = np.array(keys, dtype=float, copy=True)
        snapshot.flags.writeable = False
        return self.decode(snapshot, self.instance)

    def cost(self, solution: object) -> float:
        """Evaluate a solution independently and retain bounded best-witness evidence.

        Args:
            solution (object): Candidate-produced decision indices.

        Returns:
            float: Fixed feasibility-first minimization score.

        Raises:
            ValueError: If output is structurally invalid.
        """
        evidence = self.problem.evaluate(solution)
        self.evaluations += 1
        self.infeasible += not evidence["feasible"]
        indices = cast(Iterable[int], solution)
        if len(self.distinct) < 256:
            self.distinct.add(tuple(indices))
        if self.best is None or evidence["score"] < self.best["score"]:
            self.best = {**evidence, "solution": list(indices)}
        return float(evidence["score"])

    def _rko(self, seed: int) -> None:
        """Execute the unchanged fixed Python sequential SA configuration.

        Args:
            seed (int): Solver random seed.

        Raises:
            Exception: If candidate callbacks or the optimizer fail unexpectedly.
        """
        solver = RKO(
            self.decoder,
            self.cost,
            self.problem.dimension,
            algorithms=["SA"],
            adaptation="fixed",
            engine="python",
            execution="sequential",
            prefill_pool=False,
            pool_size=4,
            random_state=seed,
        )
        try:
            solver.solve(time_limit=max(1e-9, self.deadline - time.perf_counter()))
        except _InnerDeadline:
            pass

    def _control(self, mode: str, seconds: float, rng: np.random.RandomState) -> None:
        """Execute the fixed check probes or the matched-runtime controls.

        Args:
            mode (str): Check, random, fixed or constructive mode.
            seconds (float): Inner control budget.
            rng (np.random.RandomState): Seeded per-job random stream.

        Raises:
            ValueError: If the mode is unknown or a solution is invalid.
            Exception: If generated decoder execution fails.
        """
        dimension = self.problem.dimension
        if mode == "check":
            probes = [
                np.zeros(dimension),
                np.ones(dimension),
                np.full(dimension, 0.5),
                np.linspace(0, 1, dimension),
                rng.random(dimension),
            ]
            for keys in probes:
                self.cost(self.decoder(keys))
        elif mode in {"random", "fixed", "constructive"}:
            fixed = np.full(dimension, 0.5)
            while self.best is None or time.perf_counter() - self.started < seconds:
                if mode == "constructive":
                    self.cost(self.problem.constructive())
                else:
                    try:
                        self.cost(self.decoder(fixed if mode == "fixed" else rng.random(dimension)))
                    except _InnerDeadline:
                        break
        else:
            raise ValueError("Unknown evaluation mode.")

    def execute(self, job: dict[str, Any]) -> dict[str, Any]:
        """Time execution and assemble the bounded final evidence record.

        Args:
            job (dict[str, Any]): Mode, duration and seed settings.

        Returns:
            dict[str, Any]: Worker witness and resource measurements.

        Raises:
            Exception: If mode execution fails or no witness was produced.
        """
        self.started = time.perf_counter()
        mode, seconds, seed = job["mode"], job["seconds"], job["seed"]
        self.deadline = self.started + seconds
        rng = RandomStreams(seed).numpy
        if mode == "rko":
            self._rko(seed)
        else:
            self._control(mode, seconds, rng)
        return {
            "status": "ok",
            **cast(dict[str, Any], self.best),
            "evaluations": self.evaluations,
            "infeasible_evaluations": self.infeasible,
            "distinct_outputs_capped": len(self.distinct),
            "wall_seconds": time.perf_counter() - self.started,
            "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
            + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }


def evaluate_job(job: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one job inside a qualified, externally restricted container.

    Args:
        job (dict[str, Any]): Trusted controller job with validated source/instance.

    Returns:
        dict[str, Any]: Final independently checkable witness and worker measurements.

    Raises:
        ValueError: If source, output or settings violate the job contract.
        Exception: If the decoder or unchanged optimizer fails.
    """
    return _JobEvaluation(job).execute(job)


def main() -> None:
    """Dispatch a single worker job and emit bounded success/failure JSON.

    Raises:
        ValueError: If the job envelope exceeds its size limit or is invalid JSON.
    """
    payload = sys.stdin.buffer.read(65537)
    if len(payload) > 65536:
        raise ValueError("Worker input exceeds 64 KiB.")
    job = json.loads(payload)
    if "probe" in job:
        print(json.dumps(_probe(job["probe"])))
        return
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = evaluate_job(job)
    except Exception as error:
        result = {
            "status": "error",
            "error": f"{type(error).__name__}: {str(error)[:300]}",
            "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
            + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        }
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
