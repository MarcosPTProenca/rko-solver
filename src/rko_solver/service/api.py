"""Optional authenticated FastAPI application factory for one bounded Linux service.

Objective:
- Connect existing reader/decoder/objective/checker code without changing core RKO APIs.

Technical approach:
- Validate bounded JSON against installed schemas before supervisor admission.
- Use ordinary sync HTTP handlers in the framework threadpool, never callback CPU work.
- Require a deployment-injected bearer secret and expose only fixed attachment names.

Flow (pseudocode):
1. Bind trusted adapter, secret and server caps at application startup.
2. Own supervisor startup/shutdown through FastAPI lifespan.
3. Accept one job, poll its safe result or request real process-group cancellation.
4. Download bounded result/report attachments before TTL/count expiration.

Important criteria:
- Core installation never imports FastAPI; this module belongs to the service extra.
- This is private single-tenant infrastructure, not an internet-facing sandbox/SaaS.
- API code never accepts Python/import names/pickle/host paths or arbitrary worker settings.

Inputs/Outputs:
- Input: installed typed adapter, required secret, private storage and server caps.
- Output: a FastAPI application using separately supervised callback processes.
"""

from .boundary import RequestBoundary
from .manager import AdmissionError
from .manager import JobManager
from .models import JobSubmission
from .models import ServiceAdapter
from .models import ServiceLimits
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from importlib.util import find_spec
from pathlib import Path
from pydantic import ValidationError
from uuid import UUID


def create_app(
    adapter: ServiceAdapter,
    *,
    api_key: str,
    limits: ServiceLimits | None = None,
    storage_directory: Path = Path("/tmp/rko-service"),
) -> FastAPI:
    """Bind trusted application code to an authenticated bounded optional service.

    Args:
        adapter (ServiceAdapter): Installed bounded schema/importable callback factory.
        api_key (str): Required 32–256-character non-whitespace ASCII bearer secret.
        limits (ServiceLimits | None): Server caps; None uses bounded reference defaults.
        storage_directory (Path): Dedicated private owner-controlled directory.

    Returns:
        FastAPI: Application with health, schema, submit/status/cancel/download endpoints.

    Raises:
        ValueError: If the secret, adapter spawn binding or optional report setup is invalid.
        RuntimeError: At lifespan startup if Linux/private single-worker storage is unavailable.
    """
    if (
        not isinstance(api_key, str)
        or not 32 <= len(api_key) <= 256
        or not api_key.isascii()
        or not api_key.isprintable()
        or any(value.isspace() for value in api_key)
    ):
        raise ValueError("Provide a server-generated 32–256-character ASCII bearer secret.")
    policy = limits if limits is not None else ServiceLimits()
    if policy.allow_reports and find_spec("plotly") is None:
        raise ValueError(
            "Enabled reports require the independently optional rko-solver[reports] extra."
        )
    manager = JobManager(adapter, policy, storage_directory)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Start supervision and confirm owned job termination before application shutdown.

        Args:
            app (FastAPI): Bound service application.

        Returns:
            AsyncIterator[None]: One active application lifetime.

        Raises:
            RuntimeError: If startup ownership or shutdown confirmation fails.
        """
        manager.start()
        try:
            yield
        finally:
            manager.close()

    app = FastAPI(
        title="RKO private job service",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.job_manager = manager
    app.add_middleware(RequestBoundary, api_key=api_key, max_body=policy.max_body_bytes)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Redact Pydantic input values and application validator exception messages.

        Args:
            request (Request): Rejected request, deliberately not logged/returned.
            exc (RequestValidationError): Rejected validation details, deliberately not echoed.

        Returns:
            JSONResponse: Static schema-validation error without private input values.
        """
        return JSONResponse(
            status_code=422, content={"detail": "Invalid request; see authenticated /schema."}
        )

    @app.get("/health")
    def health() -> JSONResponse:
        """Report supervisor readiness without optimization or private configuration.

        Returns:
            JSONResponse: Cheap status/incarnation probe, or 503 after supervisor failure.
        """
        return JSONResponse(
            status_code=503 if manager.failed else 200,
            content={
                "status": "unavailable" if manager.failed else "ok",
                "instance_id": manager.instance_id,
                "storage": "ephemeral",
            },
        )

    @app.get("/schema")
    def schema() -> dict[str, object]:
        """Describe installed public JSON structure without exposing its Python import target.

        Returns:
            dict[str, object]: Submission and instance schemas plus selected public caps.
        """
        return {
            "submission": JobSubmission.model_json_schema(),
            "instance": adapter.instance_model.model_json_schema(),
            "caps": {
                "body_bytes": policy.max_body_bytes,
                "variables": policy.max_variables,
                "algorithms": policy.max_algorithms,
                "search_seconds": policy.max_search_seconds,
                "wall_seconds": policy.max_wall_seconds,
                "records": policy.max_records,
                "retention_seconds": policy.retention_seconds,
                "reports": policy.allow_reports,
            },
        }

    @app.post("/jobs", status_code=202)
    def submit(body: JobSubmission) -> dict[str, object]:
        """Validate application structure/settings before allocating an optimizer process.

        Args:
            body (JobSubmission): Byte/depth-bounded finite JSON from an authenticated client.

        Returns:
            dict[str, object]: Accepted server job UUID and current lifecycle state.

        Raises:
            HTTPException: With 422 for invalid schemas/settings or 429 while busy.
        """
        try:
            body.options.checked(policy)
            parsed = adapter.instance_model.model_validate(body.instance)
            body.instance = parsed.model_dump(mode="json")
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, "Invalid instance or search settings; see /schema.") from exc
        try:
            identifier = manager.submit(body)
        except AdmissionError as exc:
            raise HTTPException(
                429, "Service busy; no waiting queue.", headers={"Retry-After": "1"}
            ) from exc
        return manager.view(identifier)

    @app.get("/jobs/{job_id}")
    def status(job_id: UUID) -> JSONResponse:
        """Poll explicit state without claiming feasibility, resume or durable recovery.

        Args:
            job_id (UUID): Server-generated opaque job identifier.

        Returns:
            JSONResponse: Lifecycle/result metadata, or 410 for a retained expiration tombstone.

        Raises:
            HTTPException: With 404 for unknown/erased IDs.
        """
        try:
            data = manager.view(str(job_id))
        except KeyError as exc:
            raise HTTPException(
                404, "Unknown or expired job for this service incarnation."
            ) from exc
        return JSONResponse(status_code=410 if data["state"] == "expired" else 200, content=data)

    @app.delete("/jobs/{job_id}", status_code=202)
    def cancel(job_id: UUID) -> dict[str, object]:
        """Request group termination; a 202 is not a claim that work has already stopped.

        Args:
            job_id (UUID): Known owned job identifier.

        Returns:
            dict[str, object]: Current lifecycle state; poll until terminal.

        Raises:
            HTTPException: With 404 for unknown/expired jobs.
        """
        try:
            return manager.cancel(str(job_id))
        except KeyError as exc:
            raise HTTPException(404, "Unknown or expired job.") from exc

    @app.get("/jobs/{job_id}/artifacts/{name}")
    def download(job_id: UUID, name: str) -> Response:
        """Download only bounded fixed-name attachments, never user-chosen host paths.

        Args:
            job_id (UUID): Known completed job identifier.
            name (str): Exactly result.json or report.html.

        Returns:
            Response: Bounded authenticated attachment with restrictive browser headers.

        Raises:
            HTTPException: With 404 if missing, expired, unsafe or not completed.
        """
        try:
            content = manager.artifact(str(job_id), name)
        except (KeyError, OSError) as exc:
            raise HTTPException(404, "Artifact unavailable or expired.") from exc
        return Response(
            content,
            media_type="application/json" if name == "result.json" else "text/html",
            headers={
                "Content-Disposition": f'attachment; filename="{name}"',
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "sandbox; default-src 'none'",
                "Cache-Control": "no-store",
            },
        )

    return app
