"""Exercise optional HTTP boundaries against real isolated optimizer processes.

Objective:
- Verify authentication, admission, numeric/validation separation and owned-process cleanup.

Technical approach:
- Use small importable probe callbacks with intentional blocking/failure behaviors.
- Exercise Linux spawn/session supervision through the authenticated public API.

Flow (pseudocode):
1. Bind a private schema and trusted probe factory.
2. Submit bounded requests and poll actual terminal states.
3. Assert redaction, counters, resource behavior and descendant cleanup.

Important criteria:
- Probe modes are test-only installed behavior, not public executable payloads.
- No test signals a process/group unless its PID was created by its own fixture.

Inputs/Outputs:
- Input: temporary owned directories and test-generated bearer credentials.
- Output: assertions about bounded service behavior without a Docker requirement.
"""

import json
import numpy as np
import os
import pytest
import secrets
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

try:
    pytest.importorskip("httpx2")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pydantic import BaseModel
    from pydantic import ConfigDict
    from pydantic import Field
    from pydantic import JsonValue
    from pydantic import StrictInt
    from rko_solver.assessment import ValidationCheck
    from rko_solver.assessment import ValidationReport
    from rko_solver.service import ProblemDefinition
    from rko_solver.service import ServiceAdapter
    from rko_solver.service import ServiceLimits
    from rko_solver.service import create_app
except ModuleNotFoundError as exc:
    if exc.name in ("fastapi", "pydantic", "httpx", "threadpoolctl"):
        pytest.skip("optional service/test extras are not installed", allow_module_level=True)
    raise

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux-container supervision contract"
)
TOKEN = secrets.token_urlsafe(32)
TERMINAL = {"completed", "failed", "cancelled", "timed_out", "interrupted", "expired"}


class ProbeInput(BaseModel):
    """Validate bounded test instance values without constructing an optimizer.

    Responsibilities:
    - Bound dimensions and expose only explicit installed fault-injection behaviors.

    Attributes:
        n (int): Positive dimension of at most 1024.
        mode (str): Installed test behavior, never a command or import target.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    n: StrictInt = Field(default=4, ge=1, le=1024)
    mode: Literal[
        "normal", "blocked", "spawn", "bad_cost", "check_failed", "check_error", "post_blocked"
    ] = "normal"


@dataclass
class ProbeProblem:
    """Count actual numerical callbacks and expose trusted test-only failure modes.

    Responsibilities:
    - Make objective replay and post-run reconstruction observable.

    Attributes:
        mode (str): Installed test behavior.
        calls (int): Number of objective evaluations in this job process.
        decodes (int): Number of numerical and post-run decoder calls.
    """

    mode: str
    calls: int = 0
    decodes: int = 0

    def decode(self, keys: np.ndarray) -> tuple[float, ...]:
        """Copy keys and count decoding without touching input files.

        Args:
            keys (np.ndarray): Numerical candidate.

        Returns:
            tuple[float, ...]: Immutable candidate payload.
        """
        self.decodes += 1
        return tuple(float(value) for value in keys)

    def cost(self, solution: tuple[float, ...]) -> float:
        """Evaluate a small objective or intentionally block/fail inside the owned worker.

        Args:
            solution (tuple[float, ...]): Decoded candidate.

        Returns:
            float: Finite minimization cost.

        Raises:
            RuntimeError: For the installed private-exception test mode.
        """
        self.calls += 1
        if self.mode in ("blocked", "spawn"):
            time.sleep(60)
        if self.mode == "bad_cost":
            raise RuntimeError("sensitive-instance /private/adapter.py")
        return sum(solution)

    def validate(self, solution: object) -> ValidationReport:
        """Return explicit feasibility diagnostics or a trusted checker failure/block.

        Args:
            solution (object): Post-run reconstructed candidate, not original evaluation payload.

        Returns:
            ValidationReport: Explicit passed/failed diagnostics.

        Raises:
            RuntimeError: For the installed private-checker-failure test mode.
        """
        if self.mode == "check_error":
            raise RuntimeError("sensitive-instance /private/checker.py")
        if self.mode == "post_blocked":
            time.sleep(60)
        failed = self.mode == "check_failed"
        return ValidationReport(
            (ValidationCheck("domain", "failed" if failed else "passed"),), feasible=not failed
        )

    def serialize(self, solution: object) -> JsonValue:
        """Expose callback counts without invoking the objective or decoder again.

        Args:
            solution (object): Post-run payload.

        Returns:
            JsonValue: Small diagnostic counters and reconstructed values.
        """
        return {"values": list(solution), "cost_calls": self.calls, "decode_calls": self.decodes}


def build_probe(data: dict[str, JsonValue], marker: Path) -> ProblemDefinition:
    """Build one trusted test problem and record only server-owned process markers.

    Args:
        data (dict[str, JsonValue]): Schema-validated test data.
        marker (Path): Trusted fixture path, never submitted by the HTTP client.

    Returns:
        ProblemDefinition: Bound probe callbacks and optional checker/serializer.
    """
    parsed = ProbeInput.model_validate(data)
    child_pid = None
    if parsed.mode == "spawn":
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
            ]
        )
        child_pid = child.pid
    marker.write_text(json.dumps({"pid": os.getpid(), "child": child_pid}))
    problem = ProbeProblem(parsed.mode)
    return ProblemDefinition(
        problem.decode, problem.cost, parsed.n, problem.validate, problem.serialize
    )


def make_app(root: Path, limits: ServiceLimits | None = None) -> FastAPI:
    """Bind an importable probe adapter to isolated private storage.

    Args:
        root (Path): Fixture-owned root.
        limits (ServiceLimits | None): Optional bounded test policy.

    Returns:
        FastAPI: Inert test application until TestClient enters its lifespan.
    """
    return create_app(
        ServiceAdapter(ProbeInput, partial(build_probe, marker=root / "marker.json")),
        api_key=TOKEN,
        storage_directory=root / "jobs",
        limits=limits,
    )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """Start one isolated authenticated service and shut down all owned work afterward.

    Args:
        tmp_path (Path): Pytest-owned temporary directory.

    Returns:
        Iterator[TestClient]: Authenticated application client during its lifespan.
    """
    with TestClient(make_app(tmp_path)) as current:
        current.headers["Authorization"] = f"Bearer {TOKEN}"
        yield current


def poll(client: TestClient, identifier: str, timeout: float = 15) -> dict[str, object]:
    """Wait for an actual terminal state with an independent bounded test deadline.

    Args:
        client (TestClient): Authenticated test client.
        identifier (str): Owned submitted job ID.
        timeout (float): Maximum test polling seconds.

    Returns:
        dict[str, object]: Terminal public job view.

    Raises:
        AssertionError: If the test deadline expires.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        view = client.get(f"/jobs/{identifier}").json()
        if view["state"] in TERMINAL:
            return view
        time.sleep(0.025)
    raise AssertionError("Owned job did not become terminal within the test deadline.")


@pytest.fixture
def pace(tmp_path: Path) -> float:
    """Measure how long this host currently needs to start and finish one real job.

    Wall-clock windows in the timing tests must stay longer than worker start-up, which spawns
    an interpreter and imports the optimizer; that cost grows several times over on a busy host
    while the behavior under test does not change. A healthy host measures about a second.

    Args:
        tmp_path (Path): Pytest-owned directory; the calibration app uses its own subdirectory.

    Returns:
        float: Seconds measured for one default job, never below one, used as a window multiplier.
    """
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    with TestClient(make_app(calibration)) as current:
        current.headers["Authorization"] = f"Bearer {TOKEN}"
        started = time.monotonic()
        identifier = current.post("/jobs", json={"instance": {}}).json()["id"]
        poll(current, identifier, timeout=120)
        return max(1.0, time.monotonic() - started)


def test_real_standalone_and_portfolio_preserve_counts(client: TestClient) -> None:
    """Run unchanged RKO callbacks without post-run objective replay.

    Args:
        client (TestClient): Isolated authenticated service.
    """
    for algorithms in (["SA"], ["SA", "PSO"]):
        response = client.post(
            "/jobs",
            json={
                "instance": {},
                "options": {
                    "algorithms": algorithms,
                    "time_limit": 0.05,
                    "validate_solution": True,
                    "include_solution": True,
                },
            },
        )
        assert response.status_code == 202, response.text
        view = poll(client, response.json()["id"])
        assert view["state"] == "completed", view
        result = view["result"]
        assert result["assessment"]["validation"]["status"] == "passed"
        assert result["solution"]["cost_calls"] == result["numerical"]["n_evaluations"]
        assert (
            result["solution"]["cost_calls"] + 1
            <= result["solution"]["decode_calls"]
            <= result["solution"]["cost_calls"] + 1 + len(algorithms)
        )
        assert result["assessment"]["origin"] == "post_run_decode"
        artifact = client.get(f"/jobs/{view['id']}/artifacts/result.json")
        assert artifact.status_code == 200
        assert artifact.headers["content-disposition"].startswith("attachment;")
        assert artifact.json() == result


def test_input_and_settings_rejected_before_allocation(client: TestClient, tmp_path: Path) -> None:
    """Reject invalid schemas/import/worker/tuning paths before building a job process.

    Args:
        client (TestClient): Isolated authenticated service.
        tmp_path (Path): Fixture-owned marker location.
    """
    invalid = [
        {"instance": {"n": 1000000000}},
        {"instance": {"path": "/etc/passwd"}},
        {"instance": {}, "decoder": "os.system"},
        {"instance": {}, "options": {"algorithms": ["os.system"]}},
        {"instance": {}, "options": {"algorithms": ["SA", "SA"]}},
        {"instance": {}, "options": {"n_jobs": 100}},
        {"instance": {}, "options": {"hyperparameters": {"PSO": {"p_size": 1000000000}}}},
        {"instance": {}, "options": {"time_limit": 200}},
        {"instance": {}, "options": {"report": True}},
    ]
    for body in invalid:
        response = client.post("/jobs", json=body)
        assert response.status_code == 422, response.text
    assert not (tmp_path / "marker.json").exists()
    assert client.app.state.job_manager.records == {}


def test_authentication_and_redacted_json_boundary(client: TestClient) -> None:
    """Reject unauthenticated, oversized, deep, invalid-encoding and nonfinite requests.

    Args:
        client (TestClient): Isolated authenticated service.
    """
    assert client.get("/health", headers={"Authorization": ""}).status_code == 200
    assert client.get("/schema", headers={"Authorization": ""}).status_code == 401
    assert (
        client.post(
            "/jobs", json={"instance": {}}, headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    for body in (
        b'{"instance":{"n":NaN}}',
        b'{"instance":{"n":1e9999}}',
        b'{"instance":' + b"[" * 1000 + b"]" * 1000 + b"}",
        b"\xff",
    ):
        response = client.post("/jobs", content=body, headers={"Content-Type": "application/json"})
        assert response.status_code == 400
    assert (
        client.post(
            "/jobs", content=b"x" * 65537, headers={"Content-Type": "application/json"}
        ).status_code
        == 413
    )
    rejected = client.post(
        "/jobs", json={"instance": {"sensitive-instance": "/private/customer.json"}}
    )
    assert rejected.status_code == 422
    assert "sensitive-instance" not in rejected.text and "/private" not in rejected.text


def test_busy_health_cancel_and_descendant_cleanup(tmp_path: Path) -> None:
    """Keep health responsive while cancelling a real group containing a blocked descendant.

    Args:
        tmp_path (Path): Fixture-owned marker and service storage.
    """
    with TestClient(make_app(tmp_path)) as current:
        current.headers["Authorization"] = f"Bearer {TOKEN}"
        response = current.post("/jobs", json={"instance": {"mode": "spawn"}})
        identifier = response.json()["id"]
        deadline = time.monotonic() + 10
        while not (tmp_path / "marker.json").exists() and time.monotonic() < deadline:
            time.sleep(0.025)
        marker = json.loads((tmp_path / "marker.json").read_text())
        started = time.monotonic()
        assert current.get("/health").status_code == 200
        assert time.monotonic() - started < 1
        assert current.post("/jobs", json={"instance": {}}).status_code == 429
        assert current.delete(f"/jobs/{identifier}").status_code == 202
        assert poll(current, identifier)["state"] == "cancelled"
        for pid in (marker["pid"], marker["child"]):
            status = Path(f"/proc/{pid}/status")
            try:
                status_text = status.read_text()
            except FileNotFoundError, ProcessLookupError:
                continue
            assert "State:\tZ" in status_text


@pytest.mark.parametrize("mode,expected", [("blocked", "timed_out"), ("post_blocked", "completed")])
def test_deadline_preserves_numerical_snapshot(
    tmp_path: Path, pace: float, mode: str, expected: str
) -> None:
    """Enforce a hard end-to-end deadline and keep completed numerical work after checker hangs.

    Args:
        tmp_path (Path): Fixture-owned storage.
        pace (float): Measured host start-up seconds, scaling the deadline above worker start-up.
        mode (str): Trusted blocked optimization/checker test behavior.
        expected (str): Correct numerical lifecycle terminal state.
    """
    limits = ServiceLimits(max_wall_seconds=3 * pace)
    with TestClient(make_app(tmp_path, limits)) as current:
        current.headers["Authorization"] = f"Bearer {TOKEN}"
        response = current.post(
            "/jobs", json={"instance": {"mode": mode}, "options": {"validate_solution": True}}
        )
        view = poll(current, response.json()["id"], timeout=3 * pace + 15)
        assert view["state"] == expected, view
        assert view["error"] == "job_deadline"
        if expected == "completed":
            assert view["result"]["postprocessing"]["state"] == "timed_out"
            assert view["result"]["numerical"]["n_evaluations"] > 0


@pytest.mark.parametrize("mode", ["check_failed", "check_error", "bad_cost"])
def test_numerical_validation_and_callback_failure_boundaries(
    client: TestClient, mode: str
) -> None:
    """Separate failed feasibility from private callback errors and successful numerical search.

    Args:
        client (TestClient): Authenticated isolated service.
        mode (str): Installed failure behavior.
    """
    response = client.post(
        "/jobs", json={"instance": {"mode": mode}, "options": {"validate_solution": True}}
    )
    view = poll(client, response.json()["id"])
    assert "sensitive-instance" not in json.dumps(view) and "/private/" not in json.dumps(view)
    if mode == "bad_cost":
        assert view["state"] == "failed"
    else:
        assert view["state"] == "completed"
        if mode == "check_failed":
            assert view["result"]["assessment"]["validation"]["feasible"] is False
        else:
            assert view["result"]["postprocessing"]["state"] == "failed"


def test_shutdown_restart_ttl_and_count_retention(tmp_path: Path, pace: float) -> None:
    """Stop owned work on shutdown, preserve interruption metadata and bound retained records.

    Args:
        tmp_path (Path): Dedicated private restart store.
        pace (float): Measured host start-up seconds, scaling retention above scheduling delay.
    """
    limits = ServiceLimits(max_records=1, retention_seconds=pace)
    with TestClient(make_app(tmp_path, limits)) as current:
        current.headers["Authorization"] = f"Bearer {TOKEN}"
        identifier = current.post("/jobs", json={"instance": {"mode": "blocked"}}).json()["id"]
        first_instance = current.get("/health").json()["instance_id"]
    with TestClient(make_app(tmp_path, limits)) as restarted:
        restarted.headers["Authorization"] = f"Bearer {TOKEN}"
        assert restarted.get("/health").json()["instance_id"] != first_instance
        assert restarted.get(f"/jobs/{identifier}").json()["state"] == "interrupted"
        replacement = restarted.post("/jobs", json={"instance": {}}).json()["id"]
        assert restarted.get(f"/jobs/{identifier}").status_code == 410
        assert poll(restarted, replacement)["state"] == "completed"
        time.sleep(1.2 * pace)
        assert restarted.get(f"/jobs/{replacement}").status_code == 410
        assert not (tmp_path / "jobs" / replacement).exists()


def test_one_http_worker_and_no_unrelated_storage_overwrite(tmp_path: Path) -> None:
    """Refuse duplicate supervisors and preserve unrelated preexisting storage.

    Args:
        tmp_path (Path): Fixture-owned storage.
    """
    with TestClient(make_app(tmp_path)):
        with pytest.raises(RuntimeError, match="one HTTP worker"):
            with TestClient(make_app(tmp_path)):
                pass
    other = tmp_path / "other"
    (other / "jobs").mkdir(parents=True)
    sentinel = other / "jobs" / "customer.txt"
    sentinel.write_text("do not overwrite")
    with pytest.raises(RuntimeError, match="unrelated"):
        with TestClient(make_app(other)):
            pass
    assert sentinel.read_text() == "do not overwrite"


def test_report_failure_keeps_numerical_result(tmp_path: Path) -> None:
    """Enforce an actual file-size limit without erasing completed numerical work.

    Args:
        tmp_path (Path): Fixture-owned artifacts and private storage.
    """
    pytest.importorskip("plotly")
    limits = ServiceLimits(allow_reports=True, max_artifact_bytes=65_536)
    with TestClient(make_app(tmp_path, limits)) as current:
        current.headers["Authorization"] = f"Bearer {TOKEN}"
        response = current.post("/jobs", json={"instance": {}, "options": {"report": True}})
        view = poll(current, response.json()["id"])
        assert view["state"] == "completed", view
        assert view["result"]["numerical"]["n_evaluations"] > 0
        assert view["result"]["report"]["state"] == "failed"
        assert current.get(f"/jobs/{view['id']}/artifacts/report.html").status_code == 404
        assert current.get(f"/jobs/{view['id']}/artifacts/app.py").status_code == 404
        assert current.get(f"/jobs/{view['id']}/artifacts/result.json").status_code == 200


def test_shutdown_stops_running_descendants(tmp_path: Path) -> None:
    """Confirm graceful application shutdown kills an active owned process tree.

    Args:
        tmp_path (Path): Fixture-owned process markers/storage.
    """
    with TestClient(make_app(tmp_path)) as current:
        current.headers["Authorization"] = f"Bearer {TOKEN}"
        identifier = current.post("/jobs", json={"instance": {"mode": "spawn"}}).json()["id"]
        deadline = time.monotonic() + 10
        while not (tmp_path / "marker.json").exists() and time.monotonic() < deadline:
            time.sleep(0.025)
        marker = json.loads((tmp_path / "marker.json").read_text())
    deadline = time.monotonic() + 2
    for pid in (marker["pid"], marker["child"]):
        status = Path(f"/proc/{pid}/status")
        while True:
            try:
                status_text = status.read_text()
            except FileNotFoundError, ProcessLookupError:
                break
            if "State:\tZ" in status_text:
                break
            assert time.monotonic() < deadline, f"Process {pid} survived shutdown"
            time.sleep(0.025)
    state = json.loads((tmp_path / "jobs" / identifier / "state.json").read_text())
    assert state["state"] == "interrupted"


def test_retention_expires_during_another_active_job(tmp_path: Path, pace: float) -> None:
    """Delete expired terminal artifacts even while another callback is blocked.

    Args:
        tmp_path (Path): Fixture-owned retention store.
        pace (float): Measured host start-up seconds, scaling retention above scheduling delay.
    """
    limits = ServiceLimits(retention_seconds=0.4 * pace)
    with TestClient(make_app(tmp_path, limits)) as current:
        current.headers["Authorization"] = f"Bearer {TOKEN}"
        identifier = current.post("/jobs", json={"instance": {}}).json()["id"]
        assert poll(current, identifier)["state"] == "completed"
        current.post("/jobs", json={"instance": {"mode": "blocked"}})
        time.sleep(1.2 * pace)
        assert not (tmp_path / "jobs" / identifier).exists()


@pytest.mark.parametrize("value", [False, -1, float("nan"), float("inf"), "1"])
def test_invalid_server_durations_fail_before_startup(value: object) -> None:
    """Reject malformed trusted duration policy rather than disabling supervision.

    Args:
        value (object): Deliberately invalid server configuration value.
    """
    with pytest.raises(ValueError):
        ServiceLimits(max_wall_seconds=value)
