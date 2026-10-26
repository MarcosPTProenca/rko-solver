"""Execute one trusted service job in an owned Linux process group.

Objective:
- Isolate opaque callback CPU work and preserve numerical results across post-run failures.

Technical approach:
- Establish a new session before the parent handshake authorizes application code.
- Apply CPU/file/additional-address-space limits and a one-thread numeric policy.
- Snapshot numerical JSON before optional reconstruction, validation or report rendering.

Flow (pseudocode):
1. Establish ownership/resource boundaries and acknowledge readiness.
2. Build a trusted problem once and run ordinary sequential RKO.
3. Persist the bounded numerical result before postprocessing.
4. Independently attempt optional checks/solution serialization/report generation.
5. Return only bounded JSON control messages, never client-controlled pickle/code.

Important criteria:
- The supervisor owns wall deadlines and whole-group cleanup.
- Absolute container RAM/PID/disk/CPU caps are separate from process virtual-memory limits.
- Assessment is reconstruction, not an original winning-evaluation witness.

Inputs/Outputs:
- Input: installed factory, schema-validated JSON, server limits and private server paths.
- Output: bounded result/report artifacts and small trusted IPC status messages.
"""

import json
import math
import os
import sys
import time
from ..assessment import assess_result
from ..reporting import OptimizationReport
from ..solver import RKO
from .models import JobOptions
from .models import ProblemDefinition
from .models import ServiceAdapter
from .models import ServiceLimits
from dataclasses import asdict
from multiprocessing.connection import Connection
from pathlib import Path
from pydantic import JsonValue
from threadpoolctl import threadpool_limits

if sys.platform == "linux":
    import resource


class JobWorker:
    """Run one application problem under separately supervised process ownership.

    Responsibilities:
    - Keep callback construction/optimization off the HTTP event loop.
    - Bound outputs and preserve successful numerical work if later processing fails.

    Attributes:
        None: State is owned by one job process, not a global shared optimizer.
    """

    @staticmethod
    def write_result(path: Path, data: dict[str, object], limit: int) -> None:
        """Atomically snapshot primitive finite JSON within the public-result budget.

        Args:
            path (Path): Fixed server-generated result artifact path.
            data (dict[str, object]): Numerical/post-run metadata, not arbitrary objects.
            limit (int): Maximum encoded output bytes.

        Raises:
            ValueError: If data is nonfinite, unsupported or exceeds the limit.
            OSError: If the private artifact cannot be written.
        """
        content = json.dumps(data, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(content) > limit:
            raise ValueError("Result exceeds server output policy.")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)

    @staticmethod
    def run(
        adapter: ServiceAdapter,
        instance: dict[str, JsonValue],
        options_data: dict[str, object],
        connection: Connection,
        directory: Path,
        limits: ServiceLimits,
    ) -> None:
        """Build and execute one job after the parent confirms owned-group readiness.

        Args:
            adapter (ServiceAdapter): Trusted importable construction callback/schema.
            instance (dict[str, JsonValue]): Already schema-validated JSON-safe instance.
            options_data (dict[str, object]): Previously checked numerical/post-run options.
            connection (Connection): Parent-owned control pipe, never exposed to clients.
            directory (Path): Server-generated private job directory.
            limits (ServiceLimits): Server resource/output policy.
        """
        phase = "startup"
        try:
            os.setsid()
            os.umask(0o077)
            work = directory / "work"
            work.mkdir(mode=0o700)
            os.chdir(work)
            cpu = max(1, math.ceil(limits.max_wall_seconds))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
            resource.setrlimit(
                resource.RLIMIT_FSIZE, (limits.max_artifact_bytes, limits.max_artifact_bytes)
            )
            status = Path("/proc/self/status").read_text()
            baseline = (
                int(
                    next(
                        line.split()[1]
                        for line in status.splitlines()
                        if line.startswith("VmSize:")
                    )
                )
                * 1024
            )
            address_limit = baseline + limits.extra_address_space
            resource.setrlimit(resource.RLIMIT_AS, (address_limit, address_limit))
            connection.send_bytes(b"ready")
            if connection.recv_bytes(16) != b"run":
                return
            phase = "instance"
            options = JobOptions.model_validate(options_data).checked(limits)
            with threadpool_limits(limits=1):
                problem = adapter.build(instance)
                if (
                    not isinstance(problem, ProblemDefinition)
                    or type(problem.n_variables) is not int
                    or not 1 <= problem.n_variables <= limits.max_variables
                ):
                    raise ValueError("Adapter returned an invalid/big problem definition.")
                phase = "optimization"
                solver = RKO(
                    problem.decoder,
                    problem.cost,
                    problem.n_variables,
                    algorithms=options.algorithms,
                    adaptation=options.adaptation,
                    execution="sequential",
                    pool_size=8,
                    prefill_pool=False,
                    random_state=options.random_state,
                    history_limit=32,
                )
                result = solver.solve(time_limit=options.time_limit)
                data: dict[str, object] = {
                    "numerical": asdict(result)
                    | {
                        "n_evaluations": result.n_evaluations,
                        "n_workers": result.n_workers,
                        "stop_reason": result.stop_reason,
                        "adaptation": result.adaptation,
                    },
                    "assessment": None,
                    "solution": None,
                    "postprocessing": {"state": "not_requested", "elapsed_seconds": 0.0},
                    "report": {"state": "not_requested"},
                }
                target = directory / "result.json"
                JobWorker.write_result(target, data, limits.max_json_bytes)
                connection.send_bytes(b"numerical")
                phase = "postprocessing"
                started = time.monotonic()
                assessment = None
                if options.validate_solution or options.include_solution:
                    try:
                        assessment = assess_result(
                            result,
                            decoder=problem.decoder,
                            validator=problem.validator if options.validate_solution else None,
                        )
                        data["assessment"] = assessment.to_dict()
                        if options.include_solution:
                            if problem.serialize_solution is None:
                                raise ValueError("Application solution serializer is unavailable.")
                            data["solution"] = problem.serialize_solution(assessment.solution)
                        data["postprocessing"] = {"state": "completed", "origin": "post_run_decode"}
                    except Exception:
                        data["solution"] = None
                        data["postprocessing"] = {
                            "state": "failed",
                            "code": "postprocessing_failed",
                        }
                data["postprocessing"]["elapsed_seconds"] = time.monotonic() - started
                JobWorker.write_result(target, data, limits.max_json_bytes)
                if options.report:
                    rendered = time.monotonic()
                    data["report"] = {"state": "running"}
                    JobWorker.write_result(target, data, limits.max_json_bytes)
                    try:
                        report = OptimizationReport(
                            result, assessment=assessment, decoder_diagram=problem.diagram
                        )
                        report_path = report.write_html(directory / "report.html")
                        if report_path.stat().st_size > limits.max_artifact_bytes:
                            raise ValueError("Report exceeds artifact policy.")
                        data["report"] = {"state": "completed", "artifact": "report.html"}
                    except Exception:
                        data["report"] = {"state": "failed", "code": "report_failed"}
                        (directory / "report.html").unlink(missing_ok=True)
                    data["report"]["elapsed_seconds"] = time.monotonic() - rendered
                    JobWorker.write_result(target, data, limits.max_json_bytes)
                connection.send_bytes(b"done")
        except BaseException:
            try:
                connection.send_bytes(json.dumps({"error": f"{phase}_failed"}).encode())
            except OSError, EOFError:
                pass
        finally:
            connection.close()
