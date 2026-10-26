"""Probe the clean reference container and verify its actual resource/security settings.

Objective:
- Exercise installed standalone/portfolio/heavier-input/report paths under explicit Docker caps.

Technical approach:
- Use standard-library authenticated HTTP and inspect only allowlisted container fields.
- Check non-root/read-only behavior and cgroup peak memory without exposing environment secrets.

Flow (pseudocode):
1. Wait for the container's cheap health probe.
2. Reject unauthenticated/invalid input and run three bounded trusted scheduling jobs.
3. Verify responsive health, serial numerical output and offline report attachment.
4. Assert actual Docker CPU/RAM/PID/tmpfs/read-only/init/local-port policy.
5. Print only scoped calibration sizes/peak memory, never credentials/input values.

Important criteria:
- This is calibration of the reference scheduling adapter, not arbitrary-adapter certification.
- The container identifier is CI/operator input; API clients never select Docker resources.

Inputs/Outputs:
- Input: running reference container, local URL and environment-injected API secret.
- Output: assertions and bounded nonsecret calibration observations.
"""

import argparse
import json
import os
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import Request
from urllib.request import urlopen


class ContainerSmoke:
    """Verify the private service from outside its clean constrained Linux container.

    Responsibilities:
    - Test installed callback execution and actual Docker restrictions together.

    Attributes:
        base (str): Operator-selected local service URL.
        token (str): Environment-injected ephemeral secret, never logged.
    """

    def __init__(self, base: str) -> None:
        """Bind a local endpoint and required deployment/test secret.

        Args:
            base (str): Local service base URL.

        Raises:
            KeyError: If the deployment secret is not injected.
        """
        self.base, self.token = base.rstrip("/"), os.environ["RKO_SERVICE_API_KEY"]

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        expected: int = 200,
        authenticated: bool = True,
    ) -> bytes:
        """Send bounded JSON using only trusted local endpoint paths.

        Args:
            method (str): HTTP method.
            path (str): Operator/test-controlled endpoint path.
            body (dict[str, object] | None): Small synthetic instance/options.
            expected (int): Expected HTTP status.
            authenticated (bool): Whether to supply the ephemeral bearer header.

        Returns:
            bytes: Bounded response body.

        Raises:
            AssertionError: If status/output size violates the reference contract.
            OSError: If the local service is unavailable.
        """
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(
            self.base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
            method=method,
        )
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as exc:
            response = exc
        with response:
            assert response.status == expected, (path, response.status)
            content = response.read(8_000_001)
            assert len(content) <= 8_000_000
            if path.endswith("report.html"):
                assert "attachment" in response.headers["Content-Disposition"]
                assert "sandbox" in response.headers["Content-Security-Policy"]
            return content

    def job(self, dimension: int, algorithms: list[str], report: bool) -> dict[str, object]:
        """Run a bounded scheduling job while verifying HTTP remains responsive.

        Args:
            dimension (int): Cheap or heavier reference instance size.
            algorithms (list[str]): Registered standalone/portfolio selection.
            report (bool): Whether to exercise offline rendering.

        Returns:
            dict[str, object]: Completed safe numerical/post-run result.

        Raises:
            AssertionError: If lifecycle, response latency or resource contract fails.
        """
        accepted = json.loads(
            self.request(
                "POST",
                "/jobs",
                {
                    "instance": {
                        "name": "container-smoke",
                        "processing_times": [2] * dimension,
                        "due_dates": [2 * dimension] * dimension,
                    },
                    "options": {
                        "algorithms": algorithms,
                        "time_limit": 0.3,
                        "validate_solution": True,
                        "include_solution": True,
                        "report": report,
                    },
                },
                expected=202,
            )
        )
        identifier = accepted["id"]
        deadline = time.monotonic() + 40
        view = accepted
        while time.monotonic() < deadline:
            started = time.monotonic()
            self.request("GET", "/health", authenticated=False)
            assert time.monotonic() - started < 2, "Health blocked behind optimizer CPU work"
            view = json.loads(self.request("GET", f"/jobs/{identifier}"))
            if view["state"] in ("completed", "failed", "timed_out"):
                break
            time.sleep(0.05)
        assert view["state"] == "completed", view.get("error")
        result = view["result"]
        assert result["numerical"]["engine"] == "python"
        assert result["numerical"]["execution"] == "sequential"
        assert result["numerical"]["n_workers"] == 1
        assert result["numerical"]["n_variables"] == dimension
        assert result["numerical"]["n_evaluations"] > 0
        assert result["assessment"]["origin"] == "post_run_decode"
        assert result["solution"] is not None
        if report:
            assert result["report"]["state"] == "completed", result["report"]
            assert (
                b"plotly"
                in self.request("GET", f"/jobs/{identifier}/artifacts/report.html").lower()
            )
        assert (
            json.loads(self.request("GET", f"/jobs/{identifier}/artifacts/result.json")) == result
        )
        return result

    @staticmethod
    def verify_container(container: str) -> None:
        """Assert actual runtime restrictions without printing private container environment.

        Args:
            container (str): Operator/CI-owned container identifier.

        Raises:
            AssertionError: If actual Docker restrictions differ from the reference policy.
            subprocess.CalledProcessError: If Docker inspection/owned-container probe fails.
        """
        inspect = subprocess.run(
            ["docker", "inspect", container], check=True, capture_output=True, text=True
        )
        data = json.loads(inspect.stdout)[0]
        host = data["HostConfig"]
        assert data["Config"]["User"] == "10001:10001"
        assert host["Init"] and host["ReadonlyRootfs"]
        assert host["Memory"] == host["MemorySwap"] == 1_073_741_824
        assert host["NanoCpus"] == 1_000_000_000 and host["PidsLimit"] == 128
        assert "ALL" in host["CapDrop"]
        assert "no-new-privileges:true" in host["SecurityOpt"]
        assert "size=64m" in host["Tmpfs"]["/tmp"]
        assert host["PortBindings"]["8000/tcp"][0]["HostIp"] == "127.0.0.1"
        probe = subprocess.run(
            [
                "docker",
                "exec",
                container,
                "python",
                "-c",
                "import os,pathlib; assert os.getuid()==10001; "
                "print(pathlib.Path('/sys/fs/cgroup/memory.peak').read_text().strip())",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        peak = int(probe.stdout.strip())
        assert 0 < peak <= host["Memory"]
        print(
            "Reference container: dimensions=4/1024; CPU=1; RAM=1GiB; PIDs=128; "
            f"memory_peak_bytes={peak}."
        )

    @staticmethod
    def main() -> None:
        """Run CI/operator probes against an already-started owned reference container.

        Raises:
            AssertionError: If startup, HTTP behavior or container restrictions fail.
        """
        parser = argparse.ArgumentParser(
            description="Probe the bounded reference service container."
        )
        parser.add_argument("--container", required=True)
        parser.add_argument("--url", default="http://127.0.0.1:8000")
        args = parser.parse_args()
        smoke = ContainerSmoke(args.url)
        deadline = time.monotonic() + 30
        while True:
            try:
                smoke.request("GET", "/health", authenticated=False)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        smoke.request("GET", "/schema", expected=401, authenticated=False)
        smoke.request("POST", "/jobs", {"instance": {"decoder": "os.system"}}, expected=422)
        smoke.job(4, ["SA"], False)
        smoke.job(4, ["SA", "PSO"], True)
        smoke.job(1024, ["SA"], False)
        ContainerSmoke.verify_container(args.container)
        print("Clean installed reference container smoke passed.")


if __name__ == "__main__":
    ContainerSmoke.main()
