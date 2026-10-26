"""Check installed service resources and a real job outside the source package.

Objective:
- Prove wheel/sdist installations contain the scaffold and execute the reference adapter.

Technical approach:
- Copy canonical resources into a private temporary project and import its trusted adapter.
- Run one authenticated TestClient job with the installed SDK and unchanged domain callbacks.

Flow (pseudocode):
1. Reject accidental source-tree SDK imports.
2. Copy installed canonical files without requiring a published release.
3. Import the generated trusted adapter and start a private application lifetime.
4. Submit/poll a real portfolio and verify numerical/validation/solution/report boundaries.

Important criteria:
- This script uses test-only HTTP client dependencies, not new core dependencies.
- Temporary input/artifacts and generated authentication remain private and disposable.

Inputs/Outputs:
- Input: isolated installed rko-solver[service,reports] plus httpx2.
- Output: assertions and a short success marker; no secrets or instance values are logged.
"""

import importlib
import rko_solver
import secrets
import sys
import time
from fastapi.testclient import TestClient
from pathlib import Path
from rko_solver.service import ServiceAdapter
from rko_solver.service import ServiceLimits
from rko_solver.service import create_app
from rko_solver.service.scaffold import ServiceScaffold
from tempfile import TemporaryDirectory


class InstalledServiceSmoke:
    """Exercise installed resources and callback execution without importing test fixtures.

    Responsibilities:
    - Catch missing packaged template data and installed spawn-import problems.

    Attributes:
        None: All runtime state belongs to one temporary invocation.
    """

    @staticmethod
    def run() -> None:
        """Run one bounded real installed portfolio with optional checks and report output.

        Raises:
            AssertionError: If packaging, job execution or output boundaries are invalid.
        """
        assert "/src/rko_solver/" not in str(Path(rko_solver.__file__).resolve()).replace("\\", "/")
        with TemporaryDirectory(prefix="rko-installed-service-") as temporary:
            root = Path(temporary)
            project = ServiceScaffold.create(root / "project")
            for name in ServiceScaffold.filenames:
                assert (project / name).is_file(), name
            sys.path.insert(0, str(project))
            adapter = importlib.import_module("adapter")
            secret = secrets.token_urlsafe(32)
            app = create_app(
                ServiceAdapter(adapter.InstanceInput, adapter.build_problem),
                api_key=secret,
                storage_directory=root / "jobs",
                limits=ServiceLimits(allow_reports=True),
            )
            with TestClient(app) as client:
                client.headers["Authorization"] = f"Bearer {secret}"
                response = client.post(
                    "/jobs",
                    json={
                        "instance": {
                            "name": "installed",
                            "processing_times": [3, 2, 5],
                            "due_dates": [3, 6, 9],
                        },
                        "options": {
                            "algorithms": ["SA", "PSO"],
                            "time_limit": 0.05,
                            "validate_solution": True,
                            "include_solution": True,
                            "report": True,
                        },
                    },
                )
                assert response.status_code == 202, response.status_code
                identifier = response.json()["id"]
                deadline = time.monotonic() + 40
                while time.monotonic() < deadline:
                    view = client.get(f"/jobs/{identifier}").json()
                    if view["state"] in ("completed", "failed", "timed_out"):
                        break
                    time.sleep(0.05)
                assert view["state"] == "completed", view.get("error")
                result = view["result"]
                assert result["numerical"]["n_workers"] == 1
                assert result["numerical"]["n_evaluations"] > 0
                assert result["assessment"]["origin"] == "post_run_decode"
                assert result["solution"] is not None
                assert result["report"]["state"] == "completed", result["report"]
                report = client.get(f"/jobs/{identifier}/artifacts/report.html")
                assert report.status_code == 200
                assert "attachment" in report.headers["content-disposition"]
                assert "sandbox" in report.headers["content-security-policy"]
                assert "plotly" in report.text.lower()
            sys.path.remove(str(project))
        print("Installed optional service/scaffold/portfolio/report smoke passed.")


if __name__ == "__main__":
    InstalledServiceSmoke.run()
