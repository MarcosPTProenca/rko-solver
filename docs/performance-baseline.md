# Baseline contracts and fixed-work diagnostics

This checklist anchors performance experiments to tested behavior. It is not a
claim that Python matches the complete C++ solver, nor a measured bottleneck
ranking. The [performance target](performance.md) remains the acceptance contract.

## Source and baseline identity

The starting revision for this investigation is
`87846d6537627e086caf64ee6e9e58685caaf816`. Its
[CI run](https://github.com/MarcosPTProenca/rko-solver/actions/runs/39790181646)
succeeded on Ubuntu and Windows, including source checks and installed wheel/sdist
smokes. The Ubuntu suite reported 956 passing tests. This is baseline correctness
and packaging evidence, not performance certification.

For every new measurement retain the actual revision, dirty state, helper/config/
source hashes and environment, not just the package version. Do not reuse the
protected historical benchmark's output or cache, or run a competing experiment
on its host while it remains active. Current-source observations must remain
separate from the historical Python `7ee93ca` comparison.

## Contracts that optimizations must preserve

Paths below are relative to `tests/`. These existing tests are executable
anchors, not an assertion that every possible behavior is exhaustively specified.

| Contract / oracle | Regression anchor | Boundary or intentional deviation |
| --- | --- | --- |
| Decoder mutation and objective-retained snapshot isolation | `rko/test_stopping.py` | Both protective copies have observable safety purposes |
| Cooperative stop checks, finite objectives and exception propagation | `rko/test_stopping.py` | An active arbitrary callback cannot be forcibly interrupted; first valid evaluation can exceed budget |
| Completed counts, assigned seeds and immutable results | `rko/test_search_metadata.py`, `rko/test_result_metadata.py` | Decoder calls and completed objectives need not be equal |
| Library-owned RNGs and callback randomness | `rko/test_rng_isolation.py` | Equal seeds across languages do not imply equal random tapes |
| Spawn, failure cleanup, stable worker seeds and managed elites | `rko/test_process_execution.py` | Native code does not automatically eliminate IPC or process startup |
| Nominal GRASP candidate/count rules and sparse extreme steps | `metaheuristics/test_grasp_line_search.py`, `metaheuristics/test_grasp_construction.py` | Exact-ratio Python integers safely extend reference bounds; no fixed-width overflow substitution |
| GRASP h and unused alpha policies | `metaheuristics/test_grasp_h_policy.py`, `metaheuristics/test_grasp_alpha_contract.py` | Preserve documented adaptive effects, not an invented alpha RCL |
| IPR candidate trace, partial blocks, signed zero and no-op pruning | `metaheuristics/test_ipr_relink.py` | Pruning deliberately differs in callback counts; first-tie/path rules remain tested |
| Nominal C++ neighborhood transitions | `components/test_cpp_operator_contracts.py` | Controlled deterministic kernel oracle, not full-driver bitstream equivalence |
| BRKGA-CS defined correlation and representative RNG | `metaheuristics/test_brkga_cs_correlation.py` | Undefined edges are excluded; rounding-fabricated constant edges are deliberately corrected, not exact-replay compatible |
| Sign-safe finite rewards and actual adaptive transitions | `components/test_reward_contracts.py` | Python deliberately corrects inherited behavior rather than reproducing defects |
| Shared initialization and optional prefill accounting | `rko/test_pool_prefill.py`, `rko/test_driver_initialization.py` | Capacity is not a seed population; prefill has a separate allowance |
| Standalone operators, legacy and public solver callers | `rko/test_solver.py`, `rko/test_rko_main_pytest.py`, `metaheuristics/test_standalone.py` | Preserve return formats and validation; scheduling backend is not a native-engine selector |

See [C++ compatibility](cpp-compatibility.md), [initialization](driver-initialization.md)
and the linked GRASP/IPR documents for the reference revision and deliberate
corrections. Remaining full-algorithm transitions, initialization, restart and
stopping equivalence require additional evidence. Test coverage is not scientific
quality evidence or a reason to skip independent objective/feasibility validation.

## Reproducible diagnostic entry point

Run from the repository root on noncompeting resources:

```bash
uv run --locked python -m benchmarks.performance_profile \
  --algorithm SA --workload sphere --dimension 20 --evaluations 64 --seed 200

uv run --locked python -m benchmarks.performance_profile \
  --algorithm SA --workload sphere --dimension 20 --evaluations 64 --seed 200 --profile
```

Both commands emit one JSON object. They reuse the existing `fixed_work` search
and its callback boundary, rather than a replacement optimizer. Supported searches
are SA, GA, BRKGA, GRASP, MultiStart and seeded IPR; workloads are sphere, rugged
and stable ordering. IPR's ten initial elite scores are reported separately from
the completed search evaluation cap. The runner restores caller Python/NumPy
random streams; use it in a standalone process, not concurrently with other RNG
users in the same process.

The record includes:

- Schema version, restricted scope, profiled flag, work/CPU/wall observations and
  result/pool/RNG replay fingerprints from the existing runner.
- Actual imported RKO/NumPy locations, Python/NumPy versions, source and bundled
  TOML hashes, lockfile/build metadata and benchmark-helper hashes.
- Git revision, status and tracked-diff hash when Git is available. Missing Git
  observations are null; source hashes remain available. This is not a complete
  archive of untracked files, installed dependencies or the execution environment.
- CPU count/affinity observations, root cgroup-v2 quota text if accessible and
  whitelisted native-thread environment settings. These do not certify effective
  hierarchical quotas, actual native teams, host isolation or reproducible load.
- With `--profile`, the top 20 functions by exclusive cProfile time, with inclusive
  time and primitive/total calls. `profile_total_seconds` retains the full
  profiler total as a denominator; the truncated top 20 are not the whole run.
  Unprofiled records use null. Inclusive values overlap: do not sum them.

`result.seconds` and `cpu_seconds` cover search only; imports, provenance, algorithm
construction, pool seeding and process startup/shutdown are excluded. Function
profiling covers the existing runner including construction and seeded-pool setup,
so its scope is explicitly different. Profiled times are diagnostic and must not
be used as acceptance timings. cProfile cannot explain all native internals, and
wall profiles are not memory measurements.

## Hosted fixed-work screening

The [baseline study workflow](https://github.com/MarcosPTProenca/rko-solver/blob/main/.github/workflows/performance-study.yml) runs on
isolated hosted Ubuntu runners and retains raw records/host observations for 90
days. It runs on relevant pull requests and can be dispatched manually after merge.
Its registered screening matrix is six diagnostic algorithms, three objective
families, dimensions 20/500, seeds 200–202 and 1,024 completed search objectives,
each plain and profiled: 216 trials / 108 replay pairs / 36 three-seed groups.
Plain/profiled order alternates by seed; profile versus plain is an equivalence
check, not a speed comparison. Output goes outside the checkout to avoid creating
a dirty source identity while collecting it.

`benchmarks.performance_report` rejects incomplete or duplicate pairs, mixed source/
runtime identities, changed result/pool/RNG fingerprints and invalid timing scopes.
It reports unprofiled search medians separately from full-runner profile medians.
Function fractions use the full profiler total, not search-only time or the sum
of the truncated list. Only functions observed in every seed's top list are ranked;
missing functions are not treated as zero. The workflow checks registered coverage.
Raw artifacts remain necessary to inspect functions excluded by this selection.

The three objective families are NOT the three callback-cost regimes in the larger
investigation plan. This is a bounded preliminary use of existing instrumentation,
not completion of Phase B. Small evaluation caps, especially GRASP at dimension500,
can observe startup/partial construction rather than a completed search stage.
Three seeds do not establish confidence bounds, quality or competitive superiority.
Profiling biases Python/native cost shares; fractions nominate questions, not
unprofiled headroom or guaranteed Amdahl gains. Different dimensions run on different
VMs, so their latency difference is not a controlled hardware-scaling experiment.

### First registered screen

[Run 39809649686](https://github.com/MarcosPTProenca/rko-solver/actions/runs/39809649686)
validated all 216 trials / 108 exact replay pairs / 36 groups. The selected compact
[reference record](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/performance-baseline-reference.json) retains input
and analysis SHA256 digests, producer revision, run URL and explicit limits. Both
raw shards had clean Git state and the same source/runtime identity: Python 3.14.9,
NumPy 2.4.3. The dimension-20 VM reported Intel Xeon Platinum 8573C; dimension 500
reported AMD EPYC 9V45. Both exposed four logical CPUs. This confirms why latency
across these dimensions must not be treated as controlled scaling evidence.

At dimension 500, the **current** shared `blending` function was consistently
visible across SA, GA and BRKGA and all three objective families: its median
exclusive **profile** fraction ranged 52.8%–60.3%. These are not unprofiled time
shares. The operator already uses batched NumPy masks; the residual per-gene
Python/RNG loop is the candidate to investigate, not a claim that its earlier
optimization remains unimplemented. Smaller dimensions showed different costs.

The next bounded question is whether that residual loop can be reduced while
preserving the ordered Python RNG draws, mutation/inheritance rules, complement
clipping, dtype/shape validation and downstream search replay. First confirm value
in unprofiled current-versus-candidate whole-search experiments, including small
vectors and expensive callbacks. Reject a microbenchmark-only win or a changed
trajectory. This screen selects a question, not an implementation, speedup or
native-backend decision. Whole-invocation, callback-regime and memory work remains.
Raw artifacts expire after 90 days; re-collect evidence before relying on it for a
later acceptance claim rather than treating this compact screen as a full archive.

### Uncommitted blending experiments

The same workflow accepts an optional `candidate_source_b64` manual input: base64
of a **trusted** Python module exporting `blending(a, b, factor)`. This executes
arbitrary Python, not a sandbox; never use unreviewed third-party source. The job
has read-only repository permissions and does not persist checkout credentials.
An empty input retains the ordinary baseline screen. A candidate is transmitted
through an environment variable, never interpolated into shell/Python source.

The experimental job uses `benchmarks.blending_work`: kernel checks compare exact
values, ownership, validation and RNG; paired real SA/GA/BRKGA searches compare
cost, keys, pool, completed work and RNG fingerprints. Only Nelder-Mead's shared
blending binding changes, not GA/BRKGA's separate crossover implementations.
Bindings and caller RNG state are restored on success or failure. A deterministic
Python-heavy callback adds arithmetic but invokes the actual objective only once.
This is a controlled overhead model, not a real expensive-domain benchmark.

Defaults are dimensions20/500, five seeds starting200 and8192 completed search
objectives. Each algorithm/regime is warmed outside timing; mode order alternates.
Search-only timings exclude construction/startup. Separate same-seed operator
tracemalloc diagnostics exclude source parents and are NOT process RSS evidence.
Retained artifacts include the candidate source/hash so a failed hypothesis can
be audited without committing its code. Kernel fixtures and fixed-work screening
are not full correctness coverage or confirmation; production changes still need
independent tests, fresh confirmation seeds and whole-workflow/quality evidence.

To reproduce a trusted candidate locally on noncompeting resources:

```bash
uv run --locked python -m benchmarks.blending_work --candidate /tmp/candidate.py \
  --dimension 500 --evaluations 8192 --seeds 5 --seed-start 200
```

Do not commit candidate code or open a performance PR based on a speculative or
microbenchmark-only win. Record a failed hypothesis in the ignored plan and discard
it; the comparison runner itself is independently tested measurement capability.

This first diagnostic does not yet cover whole public-API invocations, process
portfolios, realistic expensive callbacks, stage counters, peak process-tree memory,
real-clock adaptive quality or C++ equal-resource comparisons. Follow the
[existing startup probes](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md#spawn-startup-and-installation-layout)
for separate import/spawn evidence. Do not extrapolate a tiny CI smoke to speed,
quality, a native-backend choice or release readiness.
