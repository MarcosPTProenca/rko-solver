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
from rko_solver import RKO
from types import MappingProxyType
from typing import Any


class _InnerDeadline(Exception):
    """Stop at the next callback boundary without discarding completed evidence."""


def _probe(name: str) -> dict[str, Any]:
    """Exercise PID enforcement or intentionally exhaust a resource.

    Args:
        name (str): Trusted hang/output/memory/pids probe name.

    Returns:
        dict[str, Any]: PID-enforcement evidence; other probes must not return normally.

    Raises:
        ValueError: If probe name is unknown.
        MemoryError: If the memory probe is stopped by allocation failure.
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
    artifact = DecoderArtifact.create(job["family"], job["source"])
    problem = Problem(job["family"], job["data"])
    namespace: dict[str, Any] = {}
    exec(compile(artifact.source, "decoder.py", "exec"), namespace)
    decode = namespace["decode"]
    instance = _immutable(problem.data)
    best: dict[str, Any] | None = None
    evaluations, infeasible = 0, 0
    distinct: set[tuple[int, ...]] = set()

    def decoder(keys: np.ndarray) -> object:
        """Call the candidate with a detached read-only key vector and immutable instance.

        Args:
            keys (np.ndarray): Solver-produced keys.

        Returns:
            object: Untrusted decision payload.
        """
        if best is not None and time.perf_counter() >= deadline:
            raise _InnerDeadline
        snapshot = np.array(keys, dtype=float, copy=True)
        snapshot.flags.writeable = False
        return decode(snapshot, instance)

    def cost(solution: object) -> float:
        """Independently evaluate one returned solution and retain bounded evidence.

        Args:
            solution (object): Candidate-produced decision indices.

        Returns:
            float: Fixed feasibility-first minimization score.

        Raises:
            ValueError: If output is structurally invalid.
        """
        nonlocal best, evaluations, infeasible
        evidence = problem.evaluate(solution)
        evaluations += 1
        infeasible += not evidence["feasible"]
        if len(distinct) < 256:
            distinct.add(tuple(solution))
        if best is None or evidence["score"] < best["score"]:
            best = {**evidence, "solution": list(solution)}
        return float(evidence["score"])

    started = time.perf_counter()
    mode, seconds, seed = job["mode"], job["seconds"], job["seed"]
    deadline = started + seconds
    rng = RandomStreams(seed).numpy
    if mode == "rko":
        solver = RKO(
            decoder,
            cost,
            problem.dimension,
            algorithms=["SA"],
            adaptation="fixed",
            engine="python",
            execution="sequential",
            prefill_pool=False,
            pool_size=4,
            random_state=seed,
        )
        try:
            solver.solve(time_limit=max(1e-9, deadline - time.perf_counter()))
        except _InnerDeadline:
            pass
    elif mode == "check":
        probes = [
            np.zeros(problem.dimension),
            np.ones(problem.dimension),
            np.full(problem.dimension, 0.5),
            np.linspace(0, 1, problem.dimension),
            rng.random(problem.dimension),
        ]
        for keys in probes:
            cost(decoder(keys))
    elif mode in {"random", "fixed", "constructive"}:
        fixed = np.full(problem.dimension, 0.5)
        while best is None or time.perf_counter() - started < seconds:
            if mode == "constructive":
                cost(problem.constructive())
            else:
                try:
                    cost(decoder(fixed if mode == "fixed" else rng.random(problem.dimension)))
                except _InnerDeadline:
                    break
    else:
        raise ValueError("Unknown evaluation mode.")
    return {
        "status": "ok",
        **best,
        "evaluations": evaluations,
        "infeasible_evaluations": infeasible,
        "distinct_outputs_capped": len(distinct),
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
        + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def main() -> None:
    """Dispatch a single worker job and emit bounded success/failure JSON.

    Raises:
        ValueError: If the job envelope exceeds its size limit.
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
