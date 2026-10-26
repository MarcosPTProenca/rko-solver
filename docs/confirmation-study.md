# Locked decoder artifact studies

The Self-Evolve pilot is adaptive development evidence, not confirmation. This
opt-in collector and analyzer freeze an **artifact case study** before test results
are opened. They do not establish reliability of the evolution method, authorize
production deployment, or guarantee statistical significance.

No substantial confirmation run is started automatically. Its compute allowance,
practical effect, sample sizes and power justification must be chosen separately.

## What is frozen

`rko_solver.experimental.self_evolve.study.ConfirmationStudy` requires:

- All three families: knapsack, scheduling and set cover.
- Existing `seed`, `first_shot` and selected `candidate` artifacts per family.
  First-shot means the first valid modification of that seed, including a
  screen-rejected source. It receives a **complete** test panel, not cascade censoring.
- Fresh exact-reference instances, unique within the study and disjoint from the
  supplied hashes of **all** previously exposed development/validation/smoke cases.
  Hash exclusion catches exact overlap, not every semantic near-duplicate; the
  operator remains responsible for genuinely fresh sampling.
- A declared solver-seed panel, inner runtime, pinned image, CPU/elapsed limits,
  practical threshold in **absolute bounded-loss units**, and analysis settings.
- Optional size/structure stratum labels: at least two independent cases per
  stratum. Resampling preserves their fixed weights. Omission declares a single
  mixture stratum, not an undocumented stratified design.

The `transport` declaration is `per_job` by default or explicitly `persistent_exec`.
The latter reuses a leased rootless container, but starts a **fresh supervised Python
process and private temporary directory for every job**. It requires qualified GNU
timeout and init reaping; there is no fallback if those controls fail. Source file
I/O remains forbidden, and the container's bounded writable filesystem is shared
between jobs: this is not a stronger hostile-code security claim. Any failed exec
retires the entire lease and requires explicit requalification before more work.
Container leases expire automatically after controller loss. Startup charges are
recorded once per qualified session rather than hidden or repeated for every job.

The source bytes, problem data/references, randomized execution order, collector
code, NumPy/Python versions and analysis settings are hash-bound. An old export
bound to another image is rejected unless runtime rebinding is explicitly requested.
Rebinding copies **unchanged source**, records the original image and freezes the
new runtime; it does not relabel old measurements as new evidence.

`claim="exploratory"` is the default. A `confirmatory` declaration additionally
requires an external `power_plan` with a nonempty `method`, `planned_power >= 0.8`
and at least ten `calibration_instances`. This is an operator declaration requiring
scientific review—not a power calculation or certification performed by this tool.
Use independent calibration, not a convenient post-test explanation.

## API and design file

Create artifacts through the existing non-executing `DecoderArtifact.load`. Supply
`artifacts[family][role]` and `instances[family]` explicitly; do not ask the collector
to generate a missing decoder or silently choose a budget.

<!-- doc-test: skip Requires caller-supplied private instances, frozen artifacts and owner-approved budgets. -->
```python
from rko_solver.experimental.self_evolve.study import ConfirmationStudy

study = ConfirmationStudy.create(
    output_directory,
    artifacts=frozen_artifacts,
    instances=fresh_test_instances,
    prior_instance_hashes=all_previously_exposed_hashes,
    image=pinned_image_id,
    seeds=declared_solver_seeds,
    seconds=declared_inner_seconds,
    cpu_budget=approved_evaluation_cpu_seconds,
    elapsed_budget=approved_elapsed_seconds,
    minimum_effect=declared_absolute_loss_reduction,
    strata=declared_strata,
    claim="exploratory",
)
print(study.digest, study.progress())
```

All names in this example are caller-supplied values, not implicit defaults or
approval to spend those resources. `ConfirmationStudy.instance_hash(problem)`
produces identities for development and test manifests.

The CLI accepts the same declaration in JSON. `artifacts` contains paths relative
to the design file, and `instances` contains raw family data dictionaries. Other
keys match `create`, excluding `path` and `allow_runtime_rebind` (the latter is the
explicit CLI flag). This file may contain private instance data; keep it outside
proposer contexts.

```bash
uv run python -m rko_solver.experimental.self_evolve study-create design.json /private/study
uv run python -m rko_solver.experimental.self_evolve study-progress /private/study
uv run python -m rko_solver.experimental.self_evolve study-run /private/study \
  --approve-spec EXACT_STUDY_SHA256 --max-jobs 12
uv run python -m rko_solver.experimental.self_evolve study-report /private/study
```

Use `study-create --rebind-runtime` only when deliberately evaluating unchanged
old source under a new declared runtime. Study execution uses its frozen image;
there is no runtime override at collection time. Actual execution requires live
rootless qualification before every CLI collection chunk. Use the study-owned
runtime lifecycle for Python collection too:

<!-- doc-test: skip Requires an owner-approved frozen study and qualified Linux rootless runtime. -->
```python
with study.runtime(approved_spec=owner_approved_study_digest) as runner:
    study.run(runner, approved_spec=owner_approved_study_digest)
print(study.progress())
```

The lifecycle durably reserves **900 seconds before constructing or qualifying**
the runtime, including a sixty-second cleanup reserve. This deliberately
pessimistic admission bound covers bounded administrative calls and the complete
qualification probe sequence; it is not a startup forecast or extra allocation.
The remaining CPU and elapsed budgets must fit that reserve plus another job.
Known completed setup/cleanup is charged by its measured conservative wall bound;
cleanup is never truncated to its reserve, and an overrun invalidates the runtime.
Unknown/failed setup retains its reserve and permanently stops this panel.
Qualification records remain in `qualifications.jsonl`; their setup charge is
part of the declared allocation. Already exhausted/invalid panels cannot launch
another runtime. Real runners passed directly without this lifecycle are rejected.

## Collection and failure policy

Every paired instance/solver-seed block contains all six arms: seed RKO, independent
constructive baseline, first-shot RKO, candidate RKO, candidate random-key sampling
and candidate fixed-key decoding. Block order and within-block arm order are
randomized **before** collection. All use the same fixed serial SA/configuration
where RKO applies; controls use the same runtime/resource allowance.

The conservative grace reservation covers exec launch, guardian and cleanup delays;
failed/interrupted jobs retain it. Startup is charged separately for persistent
sessions, including requalification restarts, and never refunded as free compute.

SQLite commits a conservative reservation before launching each job and replaces
it with the charged upper bound only after a complete result. Known consumption is
never clipped to the reservation: an overrun retains its full charge, disables the
runtime and durably blocks further collection and inference as an infrastructure
error, even when it exceeds the entire allocation. Pausing between jobs
with `--max-jobs` is resumable; the elapsed clock is not reset. Failures/timeouts
remain observations with loss one and consume their full reserve. An interrupted
controller leaves an unknown cell reserved: it is not silently retried, dropped or
invented as a successful result, and it blocks collection and inference. A stale lock requires
operator inspection, as in the development campaign.

Budget exhaustion returns incomplete progress, not a favorable partial p-value.
Runtime integrity or exact-reference contradictions durably stop the study,
including after reopening it with a newly qualified runner. Worker claims that
contradict independent witness verification are integrity errors, not ordinary
candidate quality observations. Invalid or
infeasible candidate outputs fail the strict candidate gate; constructive-control
failure also blocks passing the study's numerical checks. Source code is never
modified or replaced by the collector.

The progress estimate exposes total nominal inner allowances and the deliberately
pessimistic sum of all full reservations. Neither is a measured wall-time forecast.
Account for reserved runtime setup in the collection ledger; analysis and any
LLM discovery cost remain separate from evaluation allowances.

## Analysis and interpretation

The fixed primary loss is one for failure/no feasible solution. Otherwise, with an
independent exact optimum `f*`, use `g = abs(f - f*) / max(1, abs(f*))` and
`loss = 0.5 * g / (1 + g)`. A feasible value contradicting the reference is an error,
not an improvement to reward. This utility is **not** the optimizer's internal score
or a certified solver gap. Raw objective/feasibility measurements remain available.

Analysis runs only on a complete panel:

1. Average solver seeds **within each instance**. More seeds are not more independent
   problem instances. The inference target is conditional on the declared seed panel.
2. Bootstrap paired instance differences, within declared strata. Display marginal
   95% percentile intervals. Practical-effect checks use a Bonferroni one-sided
   lower bound over the six primary comparisons; bootstrap coverage is approximate,
   especially in small/degenerate samples.
3. Test positive baseline-minus-candidate effects with paired sign flips: enumerate
   exactly up to sixteen instances, otherwise use seeded Monte Carlo with
   `(extreme_count + 1) / (draws + 1)`, never a zero estimated p-value. The null requires
   paired-label exchangeability/symmetric instance effects; it is not assumption-free.
4. Apply Holm correction jointly to the six predeclared seed/constructive-by-family
   hypotheses. Never correct only favorable comparisons. First-shot/random/fixed
   summaries remain descriptive secondary controls, not additional corrected claims.

An exploratory report can never pass the confirmatory decision flag. Even a
confirmatory report that passes its declared numerical checks never automatically
promotes production code, proves hostile-code safety, or establishes evolution-method
reliability. Independent full evolution campaigns and a crossed/hierarchical analysis
are needed for the latter; this collector intentionally does not fake replication.

## Executed exploratory calibration

The 2026-10-25 replacement panel completed **540/540 cells**, with fifteen
independent cases per family, two declared solver seeds and no interrupted cells.
The frozen specification is
`5ae8738962451ecf7735c941f538c18210bdcabf834446a5302c6274e3e0b895`;
controller `ab23207` used the explicitly qualified `persistent_exec` transport.
Cumulative conservative charges were **2346.264/2700 seconds**, including the
original interrupted attempt; the replacement collection window was 2655 seconds.

**The original numerical report is not valid inferential evidence.** Subsequent
pre-upload investigation recovered Podman journal timestamps showing roughly
24.5/25.2 seconds before the failing exec events, against a 25.5-second host deadline.
Those outer transport timeouts had been misattributed as candidate failures. Raw
cells and the original report remain preserved, not repaired by dropping/retrying
cells. The historical scheduling/set-cover numerical passes must not be cited as
validated improvement. Descriptively, random-key means were lower than candidate
RKO in every family; no optimizer contribution or promotion is established.

The corrected leased transport disables qualification on a host I/O timeout and
stops collection as an infrastructure error, blocking inference. Qualification
probes cannot count a host timeout as proof of worker enforcement. Failure ledgers
retain the machine-readable reason and phase. Worker-guardian expiry is recorded
separately; it bounds the worker process, not proof of an infinite decoder loop.
No timeout limits were increased.

Durable protocol, complete raw cells, diagnostics, reports, controller packages,
runtime OCI archive and checksums are stored outside the worktree at
`~/.local/share/rko/self-evolve/calibration-2026-10-25-exec/`. The original incomplete
panel remains unchanged alongside it, without inference. Approve a fresh resource
allocation and practical effect/power design before a new sealed study;
these calibration instances are now exposed and cannot be reused as sealed tests.

## Separately frozen fresh exploratory panel

The later panel `ae4fda0487ac51fedf336689d8827454e7158db683679d40c9b3d5d950d93eaf`
completed **1080/1080 cells**, with thirty new independent cases per family, ten
cases in each fixed size stratum, and two declared solver seeds. There were no
failed or interrupted cells. It used **1778.240/2700 charged seconds**, including
prepaid setup/cleanup, within **1720.412/5400 collection seconds**. The owner
explicitly approved the new allocation and **0.001 absolute normalized-loss**
practical target before collection.

All six predeclared exploratory numerical checks passed, with the strict candidate
and constructive gates true. This is **not powered confirmation or promotion**.
Random-key controls had lower descriptive mean loss for knapsack/scheduling and
tied set cover, so added value from RKO search is still not established.

The measured controller was `d189d12`/version `0.3.0a1` in the former repository,
with image `sha256:36b2082b7a44d5b6972395dd865b7e2bf1e235ea9d2bf08754d62a34194c801c`.
These are not measurements of this standalone repository's `0.3.0a2` integration
or subsequent structure refactors. That integration has its own validation.
Exact protocol, source bundles, raw cells, complete numerical report, packages,
OCI archive and checksums are preserved at
`~/.local/share/rko/self-evolve/study-2026-10-26-prepublication/`.
The earlier invalid panel remains unchanged; none of its cells were replaced.
All these instances are now exposed, not reusable sealed tests.

Implementation references: [SciPy permutation-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)
for exact/randomized paired resampling and the conservative plus-one convention;
[SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
for paired resampling. SciPy itself is **not** an added dependency.
