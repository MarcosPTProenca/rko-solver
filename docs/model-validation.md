# Historical mathematical model validation

This page preserves an identified historical campaign, not a fresh qualification
of every current source revision. See [current model contracts](modeling.md) and
[version boundaries](installation.md).

The preview was exercised with actual **Pyomo 6.10.1, PuLP 4.2.0 and
OR-Tools MathOpt 9.15.6755** objects built from cited production, transportation,
Knapsack and assignment formulations. This goes beyond adapter-only unit tests,
but the inputs are still small public teaching/reference cases, not industrial
application certification.

## What was validated

Eight formulations run through every frontend and both scoring modes:

| Formulation | Class | Variables | Rows | Known reference |
| --- | --- | ---: | ---: | ---: |
| Mixed-product production | LP | 2 | 3 | maximize 2,600 |
| Integer-unit production | IP | 2 | 3 | maximize 2,600 |
| Production with setup cost | MILP | 3 | 4 | maximize 2,500 |
| Beer transportation, minimum demand | LP | 10 | 7 | minimize 8,600 |
| Beer transportation, exact demand | LP | 10 | 7 | minimize 8,600 |
| Public kp50 Knapsack | IP | 50 | 1 | maximize 149 |
| Five-worker/four-task assignment | IP | 20 | 9 | minimize 265 |
| Contradictory production rows | LP | 2 | 5 | reference reports infeasible |

Sources and every adaptation are documented in
`benchmarks/modeling-data/README.md`. In particular, transportation is the
continuous-flow relaxation with explicit implied finite bounds; exact-demand
transportation, integer production and setup-cost production are labelled
variants rather than claims to reproduce an upstream example unchanged.

Independent checking uses the original benchmark coefficients, **not** RKO's
compiled arrays or decoder. It checks every returned search incumbent and every
retained feasible witness: finite values, exact bounds/integrality, both row
sides, original objective and feasibility consistency. Native source structure
and stored values remain unchanged.

SciPy 1.18.1 / HiGHS provides a separately timed LP/MIP reference, checked against
known answers and hand-verifiable witnesses; Knapsack additionally has the
existing independent exact dynamic-programming oracle. This dependency is in a
benchmark-only group. **RKO conversion, decoding and optimization never call the
reference optimizer or need SciPy.** Reference witnesses are not injected into
search, used for repair or used to recover missing feasible candidates.

## Identified local reference campaign

Historical artifact: `benchmarks/references/modeling-validation-reference-13b6934.jsonl`.
These data are preserved unchanged from the qualified pre-rename implementation;
**they are not measurements of the current `rko_solver` source**. Current code has
hosted functional, installed-extra and structure/type checks; those checks do not
rerun or requalify this scientific campaign. Read the exact revision's workflow
results and perform a new source-identified campaign before making new performance claims.
It contains one source/input/runtime metadata record plus **480 run records**:

- eight models × three frontends × two scoring modes × ten seeds, 0–9;
- one common, unmodified **BRKGA**, fixed adaptation, serial execution, no prefill;
- **0.5 seconds per search**, unchanged model and objective across frontends/modes;
- default absolute row tolerance `1e-6`, relative tolerance zero;
- weighted mode uses one explicit illustrative weight **1,000**, unchanged across
  cases. It is not calibrated and is not a universal recommendation;
- 2026-10-26 UTC, Ubuntu/WSL2, Intel i7-7500U @ 2.70 GHz, four available CPUs,
  Python 3.14.8, NumPy 2.4.3;
- continuous restoration with a common **1,024-sweep** cap, no per-case tuning;
- integrated baseline HEAD `13b6934` with uncommitted implementation changes. The artifact
  hashes every actual production Python file, parameter file, benchmark helper,
  package manifest and lock; HEAD alone does **not** identify the executed source.

The campaign completed **401,157 finite evaluations** and found **zero independent
semantic-validation or execution errors**. No runs were discarded. Every group
has all ten planned seeds. These facts establish conversion/result consistency
on the tested cases, not universal search quality.

### Feasibility observed

Each cell pools the same ten-seed observations from all three frontends: **30
runs per case per scoring mode**, not 30 unrelated mathematical problem classes.

| Model | Feasibility-first | Weighted penalty |
| --- | ---: | ---: |
| Production LP | 30/30 | 30/30 |
| Integer production | 30/30 | 30/30 |
| Setup-cost MILP | 30/30 | 30/30 |
| Transportation, minimum demand | **30/30** | **30/30** |
| Transportation, exact demand | **30/30** | **30/30** |
| Knapsack kp50 | 30/30 | 30/30 |
| Assignment | 30/30 | 30/30 |
| Known contradictory production | 0/30, as expected | 0/30, as expected |

All seven feasible formulations found feasible witnesses in every run: **420/480**
feasible, with the sixty unsuccessful runs being the deliberately inconsistent
control. Exact-demand transport now succeeds in **60/60 runs**, across all three
frontends and both scorers. This is local acceptance, not a universal guarantee.

The same-source box-only control, `benchmarks/references/modeling-validation-box-reference-13b6934.jsonl`,
contains another complete 480 runs with identical models, objectives, scorers,
seeds and budgets: **314 feasible runs**, **637,306 finite evaluations**, zero
semantic/execution errors, minimum-demand transport **12/30** feasibility-first
and **2/30** weighted, and exact-demand **0/60**. It uses explicit `repair="none"`;
older code was not relabeled as this control. The original fixed-work reproduction
also changed from zero to **2,048/2,048** feasible candidates at unchanged tolerance.

Minimum-demand transportation successes, separated by frontend:

| Frontend | Feasibility-first | Weighted penalty |
| --- | ---: | ---: |
| Pyomo | 10/10 | 10/10 |
| PuLP | 10/10 | 10/10 |
| MathOpt | 10/10 | 10/10 |

Wall-clock budgets allow different completed work and stochastic trajectories;
these small samples do not establish a ranking between frontend libraries.

### Quality when feasibility was found

Signed percentage gap compares the retained original objective with the
exact-model reference, with the appropriate min/max direction. Medians are
**conditional on finding feasibility**; missing solutions have no gap, not zero.

| Model | Median gap, feasibility-first | Median gap, weighted |
| --- | ---: | ---: |
| Production LP | 0.021% | 0.021% |
| Integer production | 0.769% | 0.769% |
| Setup-cost MILP | 0.055% | 0.058% |
| Transportation, minimum demand | 29.456% | 30.839% |
| Transportation, exact demand | 10.190% | 10.190% |
| Knapsack kp50 | 5.705% | 6.376% |
| Assignment | 15.094% | 11.321% |

Some integer-production runs reached the known 2,600 reference, and a weighted
assignment run reached 265. The best observed Knapsack witness was 143, below
the reference 149. A finite feasible result is therefore not automatically
optimal. RKO does not expose an optimality certificate or certified gap.

The repaired campaign has no infeasible-root/separate-feasible-witness run; the
same-source box control has **nine**. The independent archive remains necessary
and tested. Root keys/cost are never replaced with another invocation or rebuilt
by replaying objectives.

## Timing and interpretation

Native construction, conversion/compilation, search and independent validation
are recorded separately. `result.model.time_to_first_feasible_seconds` comes from
the first successfully completed finite feasible evaluation, merged across
startup/sessions/workers. It measures from the beginning of `solve`, including
execution/startup work, but excludes prior model construction/compilation.
No-feasible-found and unknown manually constructed evidence have `None`, not a
fabricated zero or the final solve duration. Only a bounded timestamp is retained;
no candidate trajectory or objective replay is introduced.

This reference campaign used serial execution. Median observed solve duration
was 0.5037 seconds, not a hard real-time promise. There is no statistical
engine-speedup, zero-overhead, native-language or solver-superiority claim.
Separate fixed-work archive/IPC measurements are described below; they are not
included in the 480-run search-quality campaign.

Nonzero row tolerances can permit tiny signed negative differences from an
exact-model reference; such a number is not evidence of beating an optimum.
All feasibility checks retain the declared rules without hidden epsilon repair.

## Callback-boundary overhead and bounded IPC

Historical reference: `benchmarks/references/modeling-overhead-reference-13b6934.json`, produced by
`benchmarks.modeling.overhead` on the same WSL2/Python/NumPy host. This is a
component measurement, not whole-solver throughput or a general speedup claim.
It loads the exact `SearchSession` source from baseline `13b6934` into the same
interpreter and pairs identical callbacks, keys and completed counts. The
existing scheduling/Knapsack callback sources also match the baseline byte for
byte. Cheap scalar and native-array callbacks are identical between variants;
no decoder/objective was rewritten to make the candidate faster.

The final reference uses **51 paired trials**, a **0.8-second calibration target**,
randomized control/candidate order, warmed callbacks, one pinned logical CPU and
GC disabled only during each timed sample. Actual per-sample completed counts
and all timings are stored. The predeclared local protection limit is a
candidate/control ratio of **1.05**. Descriptive paired-bootstrap intervals are
not a universal statistical equivalence guarantee.

| Ordinary callback regime | Median candidate/control | Descriptive 95% interval |
| --- | ---: | ---: |
| Cheap scalar, 8 keys | 1.0135 | 0.9858–1.0471 |
| Existing Python scheduling, 32 keys | 1.0507 | 0.9953–1.0708 |
| Existing Knapsack, 50 keys | 1.0134 | 0.9759–1.0550 |
| Native numerical callback, 4,096 keys | 1.0035 | 0.9870–1.0313 |

Scalar and native-array regimes meet the local upper-bound threshold. The
scheduling and Knapsack intervals overlap it: **inconclusive, not a protection
pass and not a confirmed regression**. The integrated branch needs more isolated
performance qualification before claiming disabled-path non-regression. The
previous 71b019a-based preview met the threshold locally, but that result is not
relabeled as evidence for this changed-source integration. Both qualified source
identities were preserved; no callbacks or thresholds were tuned. Calibration
warms both paths before selecting a fixed count.

For generated models, the comparison uses the **same generated callback objects
and current boundary**, with only archive retention switched off/on by the
private diagnostic harness. These archive-isolation models use explicit box mode,
not restoration. The median enabled/control ratio is **0.9938** for 8 variables/
4 sparse rows (interval 0.9770–1.0422), and **0.9898** for 128 variables/64 sparse
rows (0.9546–1.0161). These are descriptive steady-state costs, not
zero-overhead claims. Callback construction, conversion, source extraction,
evidence-bundle assembly and result reporting are outside these timed loops.

A separate feature-cost probe evaluates 64 fixed keys per policy on original
models, with independent checks outside timing. Assignments intentionally differ;
this is **not** an identical-semantics engine speed comparison.

| Fixture | Box feasible / 64 | Repaired feasible / 64 | Box median decode (ms) | Repair median decode (ms) | Completed repair sweeps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Production LP | 47 | 64 | 0.316 | 0.352 | 17 |
| Setup MILP | 20 | 64 | 0.326 | 0.473 | 44 |
| Exact-demand transport | 0 | 64 | 0.366 | 7.773 | 9,697 |

All outputs match original-data checks. Restoration is real nonzero work and can
reduce objective evaluations under a fixed deadline. Work-limit and conditional
integer failures remain honest outcomes; box-compatible mode remains available.

The IPC probe constructs actual completed weighted-score records with distinct
infeasible search and feasible objective witnesses. It sends the same result
message as the control, plus one bounded evidence message. A spawned receiver
unpickles and acknowledges every message; no objective replay or native frontend
object is transported. Table times are medians per **logical worker-result
round**, not per evaluation or actual solver worker throughput.

| Variables / rows / worker searches | Control bytes | With evidence bytes | Control round (ms) | Evidence round (ms) |
| --- | ---: | ---: | ---: | ---: |
| 8 / 4 / 1 | 112 | 1,140 | 0.177 | 0.635 |
| 128 / 64 / 1 | 1,192 | 13,614 | 0.188 | 2.537 |
| 128 / 64 / 4 | 1,192 | 49,708 | 0.129 | 5.722 |
| 128 / 64 / 12 | 1,192 | 145,974 | 0.134 | 17.354 |

Pickle encode/decode observations are recorded separately. Receiver startup
(0.726 seconds here) and warmup are excluded from these transport timings.
Bytes use the runtime's highest pickle protocol; transport uses ordinary
`Connection.send`. A synchronous acknowledgement adds overhead and the enabled
case sends two messages, so these are not estimates of production wall time.
Retention scales with variables/rows and the bounded number of worker searches,
not candidate-history length. Very large models still require their own
memory/transport qualification; bounded does not mean free.

## Local platform qualification

`benchmarks/references/modeling-platform-reference-13b6934.json` records historical the qualified source
identities and local outcomes. Both platforms use NumPy 2.4.3 and the exact
optional frontend versions listed above; neither needs SciPy in the core.

- **Linux/WSL2, CPython 3.14.8:** 1,576 full-optional tests passed with six skips;
  the clean core passed 1,409 tests with 173 optional/host-dependent skips. Strict
  docs, Ruff/format checks, wheel/sdist build and eight isolated extra-installation
  smokes passed. The smokes cover each frontend separately and combined, both artifact
  types, serial/thread/spawn execution and fully fixed direct evaluation.
- **Native Windows 10, CPython 3.14.3:** the same full optional suite, strict docs,
  Ruff/format checks, build and all eight isolated artifact smokes passed. The
  clean NumPy-only environment explicitly confirmed frontend/SciPy/Plotly absence
  and passed 1,407 tests with 172 optional/host-dependent skips. The full native
  suite passed 1,572 tests with seven skips; a final focused run passed all22
  restoration tests, including two subsequently added write-back cases. The native
  full counts precede those cases and one box-control-reference parametrization;
  they are not relabeled as a later full run. Five skips are the separate C++
  execution tests because ordinary native wheels omit that optional build feature;
  another is missing optional `httpx2`, and one is the host's unavailable symlink
  privilege. Generated-model frontend/execution tests were not skipped. Rust wheels
  were built with an isolated SHA256-verified official Rust 1.99.1 toolchain and
  the host's existing MSVC BuildTools; no system compiler installation was needed.

These are **local functional qualification results** for the source-identified
`13b6934`-based integration, not hosted CI certification or a claim about every
supported machine. The native/release-build integration is complete, but the
disabled-path performance uncertainty above remains an explicit outstanding gate. Hosted Ubuntu/Windows CI remains configured
but unexecuted for this branch. Larger application/industrial models and the
owner's sanitized real model remain additional acceptance work. The development
preview and repository release hold remain in effect.

## Reproduce or inspect

From the repository root:

```bash
uv run --locked --group model-validation --extra pyomo --extra pulp --extra mathopt python -m benchmarks.modeling.validation --seeds 10 --time-limit 0.5 --repair continuous --repair-max-sweeps 1024 --output docs/audits/my-model-validation.jsonl
uv run --locked python -m benchmarks.modeling.validation --summarize benchmarks/references/modeling-validation-reference-13b6934.jsonl
uv run --locked --group model-validation --extra pyomo --extra pulp --extra mathopt python -m benchmarks.modeling.validation --seeds 10 --time-limit 0.5 --repair none --output docs/audits/my-box-control.jsonl
uv run --locked --group model-validation --extra pyomo --extra pulp --extra mathopt pytest tests/benchmarks/test_modeling_validation.py -q
uv run --locked python -m benchmarks.modeling.overhead --repeats 51 --target-seconds 0.8 --baseline 8917000 --output docs/audits/my-model-overhead.json
uv run --locked pytest tests/benchmarks/test_modeling_overhead.py -q
```

Output creation is exclusive: choose a new path rather than overwriting a prior
measurement. Summaries retain error/no-feasible counts and detect missing or
duplicate seeds. If an independent check or execution fails, every error is
written and the campaign exits unsuccessfully instead of reporting partial
success.

Timed outcomes depend on available work/hardware; exact objective replay is not
promised. The source, inputs, settings, seed list, failures and counts are fully
recorded for inspection. Local platform qualification does not replace hosted CI,
larger application instances or the owner's sanitized real-model acceptance.
See [model conversion contracts](modeling.md) and [current limitations](status.md).
