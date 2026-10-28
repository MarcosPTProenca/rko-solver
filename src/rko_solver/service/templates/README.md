# Private single-tenant RKO service

This is a reference, not a hardened public SaaS or an untrusted-Python sandbox.
Edit `adapter.py` to bind your installed schema, reader/construction function,
decoder, objective and optional checker. The included scheduling adapter reuses
RKO's existing example. Clients send JSON values, never Python or import paths.

## Build without publishing a release

Place one reviewed **same-version Linux-compatible RKO wheel** in `wheels/`.
The SDK must include the service API; an older PyPI release will not work.
Use the Linux wheel for your image architecture (`cp314-abi3-manylinux_2_28_x86_64` or
`..._aarch64`) of a reviewed release
or of the repository's Wheels workflow; a wheel built on macOS or Windows does not
install in this Linux image, and neither does one for another architecture. Copy it
here (or use `rko-solver-service scaffold DIR --wheel PATH` initially).

```bash
export RKO_WHEEL='rko_solver-0.3.0a4-cp314-abi3-manylinux_2_28_x86_64.whl'
export RKO_SERVICE_API_KEY="$(uv run --no-project python -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker compose build
docker compose up -d
```

Use the actual reviewed filename/version; the example is not a promised index
release. Dependencies are hash-pinned in `requirements.lock`. The reviewed official
Python base is version/digest-pinned. No source-tree mount, credentials or input
instance are baked into the image. Do not print/store `docker compose config` or
full container environment output: environment-injected credentials are secrets.

## Submit and poll

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/schema -H "Authorization: Bearer $RKO_SERVICE_API_KEY"
curl --fail http://127.0.0.1:8000/jobs \
  -H "Authorization: Bearer $RKO_SERVICE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"instance":{"name":"demo","processing_times":[3,2,5],"due_dates":[3,6,9]},"options":{"algorithms":["SA","PSO"],"time_limit":0.2,"validate_solution":true,"include_solution":true,"report":true}}'
```

Use the returned UUID with authenticated `GET /jobs/UUID` until terminal.
Authenticated `DELETE /jobs/UUID` requests cancellation (202 is not proof of
termination); poll until terminal. Download `GET /jobs/UUID/artifacts/result.json`
or `.../report.html` as attachments. HTML is offline/self-contained, not a web UI.

## Explicit operating envelope

- One HTTP worker, one isolated active job, no waiting queue; busy submissions get 429.
- Search uses sequential RKO, one numeric thread and no optional initialization allowance.
- At most three algorithms; `time_limit` is per algorithm, not the whole portfolio.
- Default cooperative cap: 2 seconds per algorithm. Independent wall deadline: 30
  seconds including startup, construction, search, assessment and rendering.
- Body: 64 KiB, depth 32, finite UTF-8 JSON only, 5-second body read deadline.
- Reference schema: 1–1024 jobs. Application schemas must bound dimensions and use
  cheap structural validators; factory/decoder/objective execute only in the job process.
- Retain at most four records/artifact directories for at most 300 seconds. Count
  pressure evicts the oldest terminal record earlier. Bounded tombstones give 410;
  older/unknown IDs give 404. Inputs are not persisted in lifecycle records.
- Numerical JSON: 256 KiB; individual artifact/file: 8 MB; temporary storage: 64 MiB.
- Compose: non-root UID 10001, read-only root, no capabilities/new privileges,
  one CPU, 1 GiB RAM with no additional swap, 128 PIDs, init/reaper enabled.
- Process CPU/file/core/additional-virtual-address-space limits supplement Docker
  caps. A 256 MiB additional virtual-space allowance is **not** an RSS/RAM guarantee.
  Hard aggregate RAM/PID/disk/CPU limits require the supplied container runtime policy.

Whole owned groups are terminated on deadline, cancellation and graceful shutdown,
including ordinary descendants. Installed code must not detach/daemonize descendants,
start processes at import time or intentionally escape its owned group. This is not
a sandbox for hostile callbacks. Keep the supplied `init: true` to reap descendants.
Container PID1 exit/replacement stops its remaining process namespace.

Same-store application restart marks unfinished records interrupted; there is no
resume. Container replacement/restart can erase tmpfs and all IDs/results; a new
health `instance_id` identifies the incarnation. There is no durable queue or
exactly-once promise. Download important results before expiry/replacement.

A completed numerical result survives checker/serializer/report failure or a
post-run deadline; inspect `postprocessing`, `report` and `error`. Failed feasibility
is a completed optimization with failed checks, not an infrastructure failure or
proof of optimality. Assessment is a separate `post_run_decode`, not the winning
original evaluation. Report generation never reruns the objective.

Limits are reference defaults, not a certificate for arbitrary adapters. Calibrate
representative cheap/heavy data and report paths before increasing limits. Linux
containers are supported; Windows/WSL hosts need a working Docker Linux engine.
For exposure beyond localhost, add trusted authentication/network authorization,
TLS/reverse-proxy policy and operational review; possession of this bearer token
gives access to **all** jobs in this single tenant. Never mount the Docker socket
or host data directories into this reference container.

Stop with `docker compose down`; ephemeral results may be lost. See the maintained
[service guide](https://marcosptproenca.github.io/rko-solver/service/) for API integration,
verification and limitations.
