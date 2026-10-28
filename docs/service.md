# Optional private FastAPI/Docker service

The service is an **optional local/private single-tenant reference**, not a public
SaaS, a hostile-Python sandbox or a production-readiness certificate. Core RKO does
not require FastAPI, Pydantic, Uvicorn or reporting dependencies.

It preserves the [application pipeline](problem-adapters.md): load/validate data
once, bind the existing decoder and objective, configure `RKO`, call `.solve(...)`,
then optionally reconstruct/check/report. Application problem classes and diagrams
remain optional for ordinary library usage.

## Start from the maintained reference

Until the reviewed service SDK is published, use its **local wheel**, not an older
PyPI package or an unmerged future release. This workflow does not upload anything:

```bash
uv sync --locked --extra service --extra reports
uv run --locked rko-solver-service scaffold ../my-private-service --wheel /absolute/path/to/reviewed/rko_solver-VERSION-cp314-abi3-manylinux_2_28_x86_64.whl
```

Use the Linux wheel (x86_64 or aarch64, matching the image architecture) of a reviewed release or of the Wheels workflow (a wheel
built on macOS or Windows cannot run in the Linux image) and its actual filename. The installed
`rko-solver-service scaffold DEST [--wheel PATH]` command copies only the canonical
`app.py`, `adapter.py`, Dockerfile, Compose configuration, hash-pinned dependency
lock and guide. `DEST` must not already exist, even if empty. No introspection,
overwrite, credential creation, downloads or container startup occurs. The optional
wheel must match the installed SDK version; no wheel archive is extracted.

In the generated project:

```bash
export RKO_WHEEL='the-exact-reviewed-wheel-filename.whl'
export RKO_SERVICE_API_KEY="$(uv run --no-project python -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker compose build
docker compose up -d
```

The reference builds from an explicitly selected local SDK wheel. Runtime
requirements are hash-pinned; the official Python 3.14.8 slim base is version- and
digest-pinned. No source-tree/host-data/Docker-socket mount or input instance is
needed. Do not bake credentials into files/images or expose rendered Compose/full
environment output. Read the generated README before customizing deployment.

## Bind trusted application code

`adapter.py` contains a cheap bounded Pydantic instance schema and an importable
`build_problem(validated_json) -> ProblemDefinition` function. The reference reuses
`SchedulingInstance`/`SchedulingProblem` from the existing example unchanged.
Already-loaded JSON needs no filesystem reader. For another domain, replace only
this schema/binding with your own trusted reader/construction logic and callbacks.

The optional service wiring is:

<!-- doc-test: skip needs the generated service project with its adapter module and the service extra -->
```python
import os
from adapter import InstanceInput
from adapter import build_problem
from rko_solver.service import ServiceAdapter
from rko_solver.service import ServiceLimits
from rko_solver.service import create_app

app = create_app(
    ServiceAdapter(InstanceInput, build_problem),
    api_key=os.environ["RKO_SERVICE_API_KEY"],
    limits=ServiceLimits(allow_reports=True),
)
```

`ProblemDefinition` binds `decoder`, `cost` and `n_variables`, plus optional
`validator`, `serialize_solution` and an explicitly supplied static `diagram`.
The factory runs once in the supervised job, never per candidate. Schemas must
bound dimensions and use cheap structural validators; they execute before process
allocation in the framework threadpool. Expensive problem construction/optimization
runs only in the isolated process. Importable schemas/factories must be
spawn-pickleable and must not start processes at module import time.

Clients cannot choose a function/module, Python source, pickle, package name,
command, host path, engine, worker count, parameter file or hyperparameter grid.
This first HTTP surface intentionally does **not** expose arbitrary scalar
hyperparameter overrides; algorithms use the installed trusted defaults. Core
Python [hyperparameter configuration](usage.md) is unchanged.

## Authenticated API

Only `GET /health` is unauthenticated; it reveals readiness/incarnation, not private
configuration. Every other endpoint requires `Authorization: Bearer TOKEN` with a
server-generated 32–256-character non-whitespace ASCII secret. Wrong/missing
credentials produce 401. No universal default credential, query-token mechanism,
CORS policy or public Swagger UI is provided. The authenticated `/schema` endpoint
describes submission/instance structure and selected public caps.

```bash
curl --fail http://127.0.0.1:8000/jobs \
  -H "Authorization: Bearer $RKO_SERVICE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"instance":{"name":"demo","processing_times":[3,2,5],"due_dates":[3,6,9]},"options":{"algorithms":["SA","PSO"],"adaptation":"fixed","time_limit":0.2,"random_state":42,"validate_solution":true,"include_solution":true,"report":true}}'
```

- `POST /jobs`: 202 plus server-generated UUID; invalid schemas/settings give 422
  **before optimizer allocation**. Only finite UTF-8 JSON is accepted; no archive,
  multipart upload or client filenames. Oversize is 413; wrong media type is 415.
- `GET /jobs/UUID`: state/result. States are `queued` (startup only), `running`,
  `completed`, `failed`, `cancelled`, `timed_out`, `interrupted` or `expired`.
- `DELETE /jobs/UUID`: 202 requests real group cancellation; poll until terminal.
  A successful HTTP request does not itself prove work already stopped.
- `GET /jobs/UUID/artifacts/result.json` or `.../report.html`: authenticated bounded
  attachments with fixed server names, `nosniff`, restrictive CSP and no-store.
  HTML is offline/self-contained, not arbitrary same-origin user HTML or a web UI.

A completed numerical result contains the unchanged `RKOResult` fields plus its
computed evaluation/worker/stop/adaptation summaries. Failed feasibility remains a
**completed optimization with failed checks**. Checker/serializer/report errors do
not erase successful numerical work. A post-run deadline/cancellation can likewise
leave `completed` numerical state with failed/interrupted postprocessing; inspect
`postprocessing`, `report` and top-level `error`, not only the lifecycle state.

Assessment is explicit `post_run_decode`, with its separate duration: it is not a
winning-evaluation witness, feasibility/optimality proof, objective replay, repair
or reranking. See [solution-validation](solution-validation.md) and
[reporting](reporting.md). Rendering is not included in timed RKO search.

## Resource and lifecycle contract

The smallest execution model is one HTTP worker and **one active isolated job,
no waiting queue**. Busy submissions receive 429/Retry-After. A private storage lock
prevents additional HTTP workers sharing storage from multiplying admission.
RKO executes `execution="sequential"`, so a portfolio never creates unrestricted inner
search workers. Numeric thread policy is set before NumPy imports in the image and
supplemented by `threadpoolctl` in the job. Ordinary Python API defaults are not
changed.

Reference defaults:

| Boundary | Limit |
| --- | --- |
| Request | 64 KiB, nesting 32, 5-second total body read |
| Instance | 1–1024 variables in the reference schema |
| Portfolio | At most 3 canonical registered algorithms |
| Cooperative search | At most 2 seconds **per algorithm**, no extra pool-prefill allowance |
| End-to-end job | 30 seconds including startup/construction/checks/rendering |
| Startup handshake | 10 seconds, also within the end-to-end deadline |
| Retention | At most 4 records/directories, at most 300 seconds |
| Numeric JSON / individual artifact | 256 KiB / 8 MB |
| Container | 1 CPU, 1 GiB RAM, no extra swap, 128 PIDs, 64 MiB tmpfs |

`ServiceLimits` configures server caps; clients cannot raise them. Process CPU,
file-size, disabled core-dump and additional-address-space limits supplement the
runtime policy. The 256 MiB **additional virtual-address-space** allowance is not
an RSS guarantee. An image/app alone cannot impose Docker CPU/RAM/PID/disk caps:
keep the supplied Compose policy. The root is read-only, UID is 10001, capabilities
are dropped, new privileges are blocked and `init: true` reaps descendants.

A supervisor outside the event loop enforces wall deadlines and terminates the
whole owned Linux group on cancellation/timeout/shutdown, including ordinary
children. Installed trusted code must not daemonize/detach children or intentionally
escape ownership. This is not isolation for hostile adapters. Keep the container
init/reaper; container PID1 exit/replacement stops its remaining process namespace.
Callback exception text, paths, input values and authentication headers are not
returned in public failures. Service logs contain job IDs/status/categories, not
raw callback exceptions; application callbacks must not print private values.

TTL is an upper bound: count pressure expires the oldest terminal output earlier.
Expiration removes artifacts, with bounded tombstones returning 410 before older
IDs become 404. Inputs are not persisted in lifecycle records. Same-store process
restart marks unfinished records `interrupted`; it never resumes optimization.
Container replacement/restart can erase tmpfs and all IDs/results: observe the new
health `instance_id`, and download important output first. There is no durable
queue, transparent recovery or exactly-once promise.

## Verification and deployment limits

The dedicated service CI builds wheel/sdist installations and a clean image without
a source mount; exercises standalone, portfolio, 4/1024-variable scheduling and
report paths; checks health responsiveness; inspects actual CPU/RAM/PID/tmpfs/
read-only/non-root/init/local-port configuration; and prints scoped peak-memory
observations. Real Linux tests cover blocked callbacks, cancellation, descendants,
shutdown, TTL/count eviction, schema/auth/privacy and numerical failure boundaries.
These are calibration cases for the reference adapter, **not a capacity certificate
for arbitrary applications**. Recalibrate heavy adapters/reports before increasing
limits. Run:

```bash
uv run --locked --extra service --extra reports --group service-test pytest tests/service -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Supported execution is a Linux container. Windows/WSL hosts need a functioning
Docker Linux engine; native Windows supervision is deliberately unsupported.
Local Python embedding remains useful for tests but does not replace container
resource/isolation policy. Beyond localhost, add TLS/reverse-proxy policy, trusted
network authorization, secret rotation and operational/security review. One bearer
secret grants access to every job in this single tenant.

Redis/Celery, Kubernetes, remote workers, multi-tenant accounts, GPU/native
acceleration and untrusted code upload remain explicitly out of scope.
