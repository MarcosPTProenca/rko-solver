# Experimental decoder evolution

Self-Evolve improves an **existing seed decoder**, leaving the RKO engine, reader,
objective, constraints and validator unchanged. It searches over source programs;
it does not train neural-network weights. This first milestone is an explicit
**Pi-assisted file exchange**, not an autonomous provider integration.

The opt-in namespace is `rko_solver.experimental.self_evolve`. Existing RKO imports and
callback workflows are unchanged. No extra Python dependency is added to the core.

## Delivered and intentionally deferred

Delivered: source/contract preflight, versioned non-pickle artifacts, three trusted
pilot problem adapters and exact bounded references, frozen constructive baselines,
rootless-container qualification, a bounded per-family campaign ledger, request
export/reviewed submission, experimental export and offline replay. A [locked-study collector and paired analyzer](confirmation-study.md)
now support the next research stage, but no substantial confirmation study or
statistical superiority claim has been completed.

Not delivered: direct API automation, automated Pi/subscription integration,
mathematical-model integration, production deployment, sealed-test statistical
confirmation, or a claim that evolved decoders outperform the baselines. Later
harness automation will use **Pi by default**, with supported existing authentication
where available, and explicit direct API access as an alternative. Subscription
eligibility/automation permissions require qualification; no silent billable fallback.

## Required seed and problem contract

Use a decoder module with `decode(keys, instance)`. A family decoder is reusable
across compatible instances: key dimension is derived from the supplied data, not
hard-coded to one training instance. The current adapters are deliberately limited:

| Family | Instance fields | Returned payload |
| --- | --- | --- |
| Knapsack | positive integer `weights`, `profits`, `capacity` | Unique selected item indices |
| Scheduling | positive integer `processing_times`, nonnegative `due_dates` | Complete job-index permutation |
| Set cover | `subsets` of zero-based elements, positive `costs`, `universe_size` | Unique selected subset indices |

Only decision indices are accepted. Trusted code reconstructs weight/profit,
completion times/tardiness and coverage/cost; candidate-supplied scores are never
trusted. Inputs are read-only in the worker. Sources may import NumPy or math,
not files, network, external solvers, ambient RNG or other RKO methods. Syntax
checks are **not** a security sandbox.

Local improvement permits at most two deterministic passes, each testing at most
`n*n+n` moves. Knapsack/set-cover repair permits at most `n` monotone removals or
additions respectively; scheduling requires valid permutations. No restarts,
convergence loops or nested optimizer. Source review must verify these bounds;
the implementation does not claim to prove arbitrary Python complexity. Submission
requires an explicit `--reviewed-local-work` attestation, in addition to execution limits.

The bundled `example-seed` command explicitly creates reference seeds for a demo.
Campaign creation never invents a seed when one is missing. In a future model
workflow, model-to-decoder conversion produces the seed **before** evolution.

## Build and qualify the runtime

Initial execution support is **Linux/WSL rootless Podman with cgroup v2**, not native
Windows. Use commands from the repository root through `uv`:

```bash
uv sync --locked --group dev
podman build --tag localhost/rko-self-evolve:dev \
  --file benchmarks/self_evolve/Containerfile src
uv run python -m rko_solver.experimental.self_evolve qualify
```

The runner pins the actual local image SHA-256; it never pulls an image during a
job. Record/archive that image for evidence-compatible replay. Changing runtime
source requires a new image and a new campaign, not a silent campaign upgrade.
Build-time package download is separate from network-disabled candidate execution.
The worker explicitly uses the Python engine with serial execution; it does not
participate in automatic engine selection. Historical `rko` bundles require their
archived controller/image; new execution uses the `rko_solver` namespace.

On WSL, the runner removes Windows-drive (`/mnt/<drive>/...`) entries from its
child-only administrative `PATH`. OCI helper discovery otherwise repeatedly
traverses Windows mounts while searching absent runtimes. Linux paths and the
operator's environment are unchanged. This does not relax isolation or extend
timeouts; any remaining host dispatch failure still stops the study.

Qualification checks effective UID, read-only root, no-new-privileges, CPU, memory,
PID and network-interface policy, and exercises infinite loop, output flood,
memory exhaustion and PID exhaustion. Cleanup failures disable the runner. Jobs
have no repository/home/socket mounts, no host credential environment, bounded
output and private bounded temporary storage. The default job ceiling is one CPU,
512 MiB memory/swap, 32 PIDs and 16 MiB temporary storage. There is no raw-process
fallback when rootless isolation is unavailable.

**Threat model:** accidental/opportunistic generated-code errors, not a guarantee
against actively hostile Python. Worker code and the evaluator share an interpreter.
The parent independently verifies final witnesses without importing candidate code,
but cannot thereby prove every trajectory/resource claim free of tampering. Source
hashes establish identity, not safety. Do not expose this as a public hostile-code
execution service.

## Pi-assisted lifecycle

For example, use a fresh directory outside the repository for each family:

```bash
uv run python -m rko_solver.experimental.self_evolve example-seed knapsack /tmp/kp-seed
uv run python -m rko_solver.experimental.self_evolve create /tmp/kp-seed /tmp/kp-campaign
uv run python -m rko_solver.experimental.self_evolve calibrate /tmp/kp-campaign
uv run python -m rko_solver.experimental.self_evolve request /tmp/kp-campaign /tmp/kp-request.json
```

Give the exported development packet to an active Pi session. Ask for a modification
of its supplied parent decoder and a short hypothesis. Keep the fixed RKO/evaluator,
data splits and budgets out of its editable scope. Do not supply held-out instances.
The user needs no **new** API key for this interactive exchange; it still uses the
session's configured provider/access and is not free callable inference. The active
Pi session remains a trusted developer session: file exchange restricts accepted
proposals, not all of that agent's tools. Hard automated proposer workspace/tool
isolation belongs to the later harness transport; do not mistake this manual mode
for an autonomously sandboxed agent.

Review the proposed local-work bounds before submission, then supply the packet's
parent hash explicitly:

```bash
uv run python -m rko_solver.experimental.self_evolve submit /tmp/kp-campaign /tmp/candidate.py \
  --parent PARENT_SHA256 --hypothesis 'Why this modification may help' --reviewed-local-work
uv run python -m rko_solver.experimental.self_evolve report /tmp/kp-campaign
```

An accepted result means **experimental shortlist**, not statistical superiority.
Endpoint/shape checks precede the frozen screen and larger development runs. Any
observed invalid/infeasible returned candidate output disqualifies the proposal;
failures remain recorded. Repair intermediate states may be infeasible. Reference
seed failures are reported, not hidden or mislabelled as valid solutions.

Request another modification using an accepted parent with `request --parent HASH`.
Rejected proposals and repairs count toward the same ten-submission limit. The
campaign stores source lineage, hypotheses, failure-inclusive JSONL measurements,
instance/solver seeds and a frozen evaluation specification. Preserve the Pi session
transcript separately for proposer/model/intervention provenance; this manual
transport does not automatically extract provider billing or conversation history.

### Frozen pilot work

Run separate sequential campaigns for knapsack, scheduling and set cover. Each has
at most ten submissions, 600 baseline/control CPU-seconds, 1,200 candidate
CPU-seconds and 5,400 elapsed seconds. The elapsed clock starts at calibration,
not artifact preparation. It continues across sessions; stopped controllers do not
reset budgets. A conservative CPU upper bound uses externally measured job wall
time plus quota-period slack, never less than reported process CPU; failures and
interrupted jobs retain their full reserved allowance. Report actual worker CPU
separately rather than calling the charged upper bound a measurement. Known cell
consumption is never clipped to its reservation; an overrun retains the full charge
and durably stops the campaign as an infrastructure error.

The initial fixed development table uses three instance sizes, two solver seeds,
0.1-second inner allowances, serial SA, fixed adaptation, pool size four and no
prefill. Runtime-startup CPU belongs to the evaluation charge; exact-reference
preparation and isolation qualification are setup costs and must be reported
separately. This is a cheap exploratory table, **not** a powered experiment. Baseline
qualification must show it fits before proposals; if useful work cannot fit, stop
and return the conflict rather than shrinking data or increasing budgets silently.

Independent controls: unchanged reference seed, ratio-greedy knapsack, ATC
scheduling (`k=2`, unit job weights, remaining-job mean processing time, index-stable
ties), cost-per-new-coverage greedy set cover, equal-runtime random-key sampling,
and repeated fixed-key decoding. The first submitted modification is also the
first-shot LLM control. Keep all raw outcomes, not just the running-best curve.

`controller.lock` prevents overlapping writers. An interrupted controller can leave
this lock intentionally: verify its recorded PID is no longer active before manual
removal. CPU reservations for interrupted jobs remain charged. An interrupted
proposal is not silently retried as a new free attempt.

## Save and replay without an LLM

```bash
uv run python -m rko_solver.experimental.self_evolve export /tmp/kp-campaign HASH /tmp/kp-export
uv run python -m rko_solver.experimental.self_evolve replay /tmp/kp-export /tmp/new-instance.json \
  --seconds 1 --seed 42 --image PINNED_IMAGE_ID
```

The instance JSON is `{"family": "knapsack", "data": {...}}`, using the contract
above. Export preserves the exact UTF-8 decoder source bytes, including line endings,
and writes a manifest and hash-bound evidence. Loading
checks source/schema/contract/runtime metadata without importing source on the host;
execution still uses a qualified isolated runner. An evidence-backed export
requires its pinned runtime image. Preserve that runtime before pruning local
images; an image ID alone does not make its bytes available on another machine:

```bash
podman save --format oci-archive --output runtime.oci.tar PINNED_IMAGE_ID
podman load --input runtime.oci.tar
```

On a compatible Linux/WSL host, replay needs neither campaign history/training data
nor API credentials/Pi at inference time. Test on a new compatible size to detect
hard-coded dimensions.

Reference seeds may be exported under their honest baseline identity even when no
evolved decoder wins. Experimental artifacts are not automatically deployed or
installed as production defaults.

## Validation and scientific claims

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest src/rko_solver/tests -q
RKO_EVOLVE_IMAGE=localhost/rko-self-evolve:dev uv run pytest src/rko_solver/tests/experimental -q
```

Live tests are opt-in; ordinary unit-test success does not certify a working
container runtime. Pilot completion permits negative outcomes and incomplete pilots
must be labelled incomplete. Meaningful superiority claims need later sealed
instance splits, predeclared practical effects, instance-clustered inference,
multiple-comparison correction and independent evolution campaigns for method-level
claims. The current development report deliberately makes no such claim.
