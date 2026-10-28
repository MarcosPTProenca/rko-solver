# Experimental decoder evolution

Self-Evolve improves an **existing seed decoder**, leaving the RKO engine, reader,
objective, constraints and validator unchanged. It searches over source programs;
it does not train neural-network weights. The manual **Pi-assisted file exchange**
remains available; an optional automated Pi proposal transport now pauses for
explicit human review before any candidate evaluation.

The opt-in namespace is `rko_solver.experimental.self_evolve`. Existing RKO imports and
callback workflows are unchanged. No extra Python dependency is added to the core.

## Delivered and intentionally deferred

Delivered: source/contract preflight, versioned non-pickle artifacts, three trusted
pilot problem adapters and exact bounded references, frozen constructive baselines,
rootless-container qualification, a bounded per-family campaign ledger, request
export/reviewed submission, experimental export and offline replay. A [locked-study collector and paired analyzer](confirmation-study.md)
now support the next research stage, but no substantial confirmation study or
statistical superiority claim has been completed.

The opt-in M5a transport targets Pi 1.3.0 on Linux/WSL and the built-in OpenAI Codex
OAuth route; it is not a live-provider qualification or a guarantee of subscription
eligibility. Not delivered: direct API automation, other automated providers,
production deployment, sealed-test statistical confirmation, or a claim that evolved
decoders outperform the baselines. The experimental M6 owned-model construction
bridge is described below; its GAP checks establish correctness, not optimizer value. Supported
account access and automation permissions require operator qualification; no silent
billable API-key fallback is provided.

## Required seed and problem contract

Use a decoder module with `decode(keys, instance)`. A family decoder is reusable
across compatible instances: key dimension is derived from the supplied data, not
hard-coded to one training instance. The three pilot adapters remain deliberately limited:

| Family | Instance fields | Returned payload |
| --- | --- | --- |
| Knapsack | positive integer `weights`, `profits`, `capacity` | Unique selected item indices |
| Scheduling | positive integer `processing_times`, nonnegative `due_dates` | Complete job-index permutation |
| Set cover | `subsets` of zero-based elements, positive `costs`, `universe_size` | Unique selected subset indices |

For these three pilots only decision indices are accepted. Trusted code reconstructs weight/profit,
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
Campaign creation never invents a seed when one is missing. Model workflows also
require an explicitly selected existing seed **before** evolution.

## Experimental mathematical construction bridge (M6)

The distinct `linear_model` artifact family uses `decode(keys, instance)` to return
**complete original-order numerical assignments**, not selected decision indices.
The immutable context contains `variables`, finite effective `domains`, original
sparse `rows`, `objective`, `objective_constant`, `maximize` and derived `n_keys`.
Fixed coordinates remain in the assignment but consume no key. The original
objective, constants, row sides, domains, tolerances and scoring remain authoritative.
Domain/shape/nonfinite failures stop the job; constructors cannot return their own score.

`ModelProblem.from_source` accepts an owned extraction from the existing Pyomo,
PuLP or MathOpt adapter, without moving native objects into workers. It uses bounded
24 KiB non-pickle JSON, explicit absent-side bounds, original source fingerprints,
effective policy and exact installed Python implementation hashes. A code/policy
change requires rebuilding the image and starting a fresh compatible campaign.
This is not cross-version migration or native-package qualification. References and
source/case names are excluded from model proposer development input; held-out data
must still be kept outside the trusted proposer workspace.

<!-- doc-test: skip Requires three caller-owned native development models and an explicitly qualified worker image. -->
```python
from pathlib import Path
from rko_solver.experimental.self_evolve.campaign import Campaign
from rko_solver.experimental.self_evolve.model_data import ModelProblem, ModelReference
from rko_solver.modeling.adapters import ModelAdapter

# Three distinct caller-owned native models and trustworthy original optimum declarations.
problems = [
    ModelProblem.from_source(
        ModelAdapter.extract(native),
        reference=ModelReference(optimum, provenance),
    )
    for native, optimum, provenance in development_models
]
seed = ModelProblem.seed_artifact()  # explicit existing domain mapper; never auto-invented
campaign = Campaign.create(
    Path("/private/model-campaign"), seed, problems, qualified_image_id,
    feasibility_policy="feasibility_first",
)
```

Original references are mandatory for automatic comparison. `ModelReference` is an
explicit owner declaration with provenance, **not** a manufactured native-solver
certificate. Unknown references are rejected before campaign state is created.
`feasibility_first` permits domain-valid row-infeasible exploration. A shortlisted
complete RKO panel must strictly improve mean primary loss, retain at least one
actual feasible witness, and preserve each case's matched-job feasibility-success
rate. Partial panels, ties and callback-throughput success fractions are not wins. `strict_feasible` rejects
any observed row-infeasible candidate output. Both keep malformed domains/protocols,
resource failures and original numerical validation strict. The three existing pilot
families retain their old strict rules. Model controls are matched RKO, random-key,
fixed-key and endpoint checks; no generic independent human baseline is invented.
The same **10 proposals / 30 CPU-minutes / 90 wall-minutes** and M5a pending-review,
exact-hash approval and failure-inclusive slot rules apply.

Model timing contract **v2** freezes a **1-second end-to-end job allowance** before
proposal feedback; the three pilot families keep their existing allowances.
Trusted model preparation, generated module initialization, construction and
protected checking all remain charged. The model adapter carries the worker's
absolute monotonic deadline into search and initialization sessions, so ordinary
RKO's startup-excluding search duration cannot renew the job budget. A fixed
**50 ms within that same second** is reserved for trusted evidence finalization
(equally for RKO/random/fixed/check controls, capped at half the allowance for
shorter offline jobs). It is not extra runtime. Parent runtime guards and the 10-proposal/30-CPU-minute/90-wall-minute ceilings are unchanged.
Older model campaign timing profiles are rejected, not silently upgraded: start a
fresh compatible campaign/image. This allowance is not a promise that every model
or arbitrarily slow callback will fit.

Construction and protected restoration/checking are charged together. The default
model bridge retains continuous repair with **1024 sweeps**, allowing explicit
`repair_sweeps` from **1 to 10000**. An actual score incumbent is kept separately
from the best feasible original-objective witness. Parent verification checks both
assignments without replaying construction, default mapping or repair. Feasible
objective, observed search score and verification score remain separate; absence of
a feasible result is explicitly `no_feasible_found`, never an optimality claim.

Export uses the ordinary reviewed campaign API/CLI. Offline `replay` accepts a JSON
`{"family": "linear_model", "data": problem.data}` for a new compatible model, with
its own derived dimension and matching protected implementation/image. Original
references are unnecessary for witness-only replay. Generated artifacts execute
only inside qualified rootless workers: never import them into the host to create
an `RKO.from_model` callback. The ordinary library's `construction=` hook accepts an
explicit **trusted callable**, not a bundle/path. Custom construction retains portable
assignments and native-ID mapping, but transactional native write-back is unsupported;
normal model write-back is unchanged. Zero-key models bypass construction/search.

The GAP fixture independently enumerates the six-job `3^6 = 729` assignments and
provides a different-size seven-job unseen case. These fixtures check original
formulations and transport/assignment correctness through all three frontends. They
are not a GAP-specific production decoder, new powered study, superiority result,
live-provider validation or production-readiness evidence.

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
observed invalid/infeasible returned candidate output disqualifies the proposal for
the three pilot families; model campaigns use their explicit frozen policy above.
Failures remain recorded. Repair intermediate states may be infeasible. Reference
seed failures are reported, not hidden or mislabelled as valid solutions.

Request another modification using an accepted parent with `request --parent HASH`.
Rejected proposals and repairs count toward the same ten-submission limit. The
campaign stores source lineage, hypotheses, failure-inclusive JSONL measurements,
instance/solver seeds and a frozen evaluation specification. Preserve the Pi session
transcript separately for proposer/model/intervention provenance; this manual
transport does not automatically extract provider billing or conversation history.

### Optional review-gated Pi transport (M5a)

This opt-in transport targets **Pi 1.3.0 on Linux/WSL**, initially with the built-in
`openai-codex` / `openai-codex-responses` route and supported Pi OAuth login. It adds
no Python dependency or default/core behavior. Other versions/providers, custom
endpoints and direct API transports are unsupported. Fake-process tests and local
metadata checks are not live authentication or provider-policy qualification.

Create a non-secret configuration outside the repository, using real absolute
paths and an exact model ID available to your eligible account:

```json
{
  "executable": "/absolute/path/to/pi",
  "profile": "/private/pi-self-evolve",
  "provider": "openai-codex",
  "model": "gpt-6-luna",
  "seconds": 120
}
```

`seconds` must be a finite number from 1 to 600. Initialize a **new** dedicated profile:

```bash
uv run python -m rko_solver.experimental.self_evolve pi-profile /private/pi-config.json
```

This creates owner-private settings, never credentials. Authenticate manually
using Pi's supported `/login` in that profile with the exact provider/model.
Do not copy `auth.json`, hand-edit credentials, supply a new API key, or weaken
an ordinary profile's settings. Use `/login` only, then exit without prompting:

```bash
PI_CODING_AGENT_DIR=/private/pi-self-evolve \
  /absolute/path/to/pi --provider openai-codex --model gpt-6-luna \
  --no-session --no-tools --no-extensions --no-mcp --no-skills \
  --no-prompt-templates --no-themes --no-context-files --no-approve --offline
```

Confirm account eligibility and provider terms yourself. OAuth is not a promise
of free generation: monetary cost is **unknown**, and provider-side work/charges
may continue after a client timeout. RKO never falls back to API-key billing.
The controller checks credential **metadata**, not credential values; it neither
copies credentials into a campaign nor sends them to decoder workers.

After creating and calibrating the ordinary seed-based campaign:

```bash
uv run python -m rko_solver.experimental.self_evolve propose /private/campaign \
  /private/pi-config.json --consent-development-data
uv run python -m rko_solver.experimental.self_evolve proposal /private/campaign 1
```

Consent authorizes sending this campaign's **development** request to the selected
provider. Keep credentials, protected evaluator internals and future sealed cases
out of seed source/development data. Generation uses a fresh private directory,
no persistent session, no tools/context discovery, fixed SSE provider transport,
no automatic retry/compaction and bounded JSON RPC input/output. Proposer settings and the controller
implementation identity are frozen before use.

Success is `pending_review`, **not evaluated or accepted**. Inspect `source_path`,
its hash and hypothesis as text; never import it on the host. The pending manifest
is intentionally not a replay/export bundle. Then approve the exact reviewed
SHA-256, or reject instead:

```bash
uv run python -m rko_solver.experimental.self_evolve approve-proposal /private/campaign 1 \
  --sha256 EXACT_REVIEWED_SOURCE_SHA256 --reviewed-local-work
uv run python -m rko_solver.experimental.self_evolve reject-proposal /private/campaign 1 \
  --reason "non-secret reason for declining the source"
```

These are alternatives, not consecutive steps. Approval reuses the reserved
proposal slot and existing isolated evaluation/acceptance rules. Export only
through the ordinary campaign command after acceptance. Approval is an operator
attestation, not proof of correctness or protection against a trusted caller
intentionally repackaging source outside this workflow. Headless Pi/process-group
supervision is **not an OS security sandbox**.

Every generation reserves one of the existing ten slots **before** Pi starts;
failed generation/rejection consumes it too. Missing usage/cost stays unknown,
never zero; available provider-reported failure usage is incomplete. The deadline
and cumulative output limits cover version/auth/RPC together, with a conservative
fifteen-second cleanup reserve. Observed overruns retain full elapsed consumption
and halt the campaign as an infrastructure error rather than clipping the charge.

Interrupted or uncertain work remains `generation_unknown`/`evaluating` and blocks
another generation. Inspect the recorded owned process groups first; only after
confirming work stopped may an operator abandon with
`reject-proposal ... --confirmed-stopped`. This refunds no slot, repairs no incomplete
evaluation reservation, certifies no provider cancellation and clears no
infrastructure halt. Preserve the private history. Controller/configuration
identity changes require a new campaign, not ledger rewriting or silent migration.

Safe requests, validated source and provenance stay owner-private; raw thinking,
unrestricted provider errors and authentication payloads are not archived. Inspect
JSON status rather than treating command completion as candidate success. This
increment includes no unattended review, modeling bridge (M6), powered
confirmation (M4), live provider qualification or added optimizer-value claim.

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
