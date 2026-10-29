# Invocation, startup and callback resource probes

The [hosted invocation study](https://github.com/MarcosPTProenca/rko-solver/blob/main/.github/workflows/invocation-study.yml) reuses
existing probes instead of implementing another solver or profiler. It measures
public `RKO` calls, preparation and resource observations, complementing the
[fixed-work diagnostics](performance-baseline.md). It does not establish a runtime
improvement or choose a native backend.

## Registered probes

Each probe runs in its own hosted Ubuntu VM. A wrapper records its source/helper
hashes, actual imported paths, runtime and native-thread environment, command,
exit status, host CPU observations and fresh probe-process elapsed time. Raw stdout,
stderr and provenance are retained for 90 days, including partial output on failure.
Each existing probe performs its own scientific/accounting assertions; the workflow
also requires the registered observation counts and complete case membership.

| Probe | Existing implementation | Registered observations |
| --- | --- | --- |
| Adaptive startup | `benchmarks/runtime/startup_stages.py --repeats 3` | Six full-portfolio solves: one/four workers, three alternating repetitions, 128 keys, online control, 0.2-second search allowance |
| Portfolio invocation/resources | `benchmarks/runtime/process_budget.py` | Twelve full-portfolio solves: sphere/ordering, seeds 42–44, automatic/ten dedicated workers, 128 keys, offline control, one-second search allowance |
| Native callback prefixes | `benchmarks/runtime/native_threads.py --repeats 3 --dimension 128 --evaluations 64 --workers 1 4` | 36 inherited/one-thread observations across sphere/GEMV/GEMM and one/four workers, plus two parent-only negative controls |

The first two probes receive a copied one-thread native environment. The callback
probe retains its inherited environment so the existing launch comparison and
parent-only negative control remain meaningful. Its worker observations verify
actual loaded native teams with benchmark-only `threadpoolctl==3.7.0`; environment
variables alone do not verify native teams. This adds no library dependency.

The callback cases deliberately stop population initialization at a fixed prefix,
not after full generations. The nominal 120-second safety allowance is not the
work performed. Three-seed/repetition diagnostics are not acceptance confidence
intervals or general quality evidence.

## Timing boundaries

- `fresh_probe_process_seconds` is the wall duration of the ENTIRE probe program
  in a fresh interpreter, including imports, orchestration, instrumentation,
  temporary-file I/O and all its solves. It is not one library invocation or pure
  `import rko_solver` latency. The OS filesystem cache is not flushed. Do not divide it
  by the solve count and call that cold per-solve startup.
- The portfolio probe's `invocation_seconds` spans `RKO(...)` construction and
  `.solve(...)` return, excluding imports, provenance collection and observation
  parsing. Its existing `elapsed_seconds` is the solver's internal elapsed field.
  Their difference includes constructor/return overhead, not a separately isolated
  constructor profile. Both clocks are unprofiled.
- Per-search metadata reports assigned allowance, elapsed time, completed objective
  count, seed and stop reason. Search intervals overlap across workers; their sum
  is not portfolio wall time or a removable overhead fraction.
- Startup construction, manager and first-callback times overlap with other work.
  Do not add inclusive stages or infer pure search time by subtracting unrelated
  maxima. Seeded first vectors and actual worker counts remain independently checked.
- Callback wall/CPU measurements exclude each worker's explicit native warm-up;
  search/invocation timing includes it. Callback process CPU includes native
  threads, not coordinator/manager CPU. Summed callback wall durations across
  workers are not a sequential fraction of end-to-end wall time.

## Memory and resource boundaries

The portfolio probe samples coordinator plus direct-child RSS approximately every
20 ms on Linux during library invocation, after probe imports. It does not measure
the interpreter-import high-water mark. Raw `rss_samples` and sampled peaks are
retained. This is not an
exact high-water mark, unique physical memory/PSS or allocation attribution; shared
pages are counted repeatedly and peaks between samples may be missed. It does not
include the outer workflow wrapper/runner or certify arbitrary nested user-created
process trees. Non-Linux memory is unavailable rather than guessed.

Worker counts, first-decode identities, loaded native teams and CPU observations
must be distinguished. The wrapper's environment is not necessarily the copied
child environment; both are recorded. Counts/affinity and root cgroup observations
do not prove all hierarchical quotas or host isolation.

Different probes run on different VMs and use different controls/budgets/work.
Do not pool their latencies, subtract timings across probes or infer controlled
hardware scaling. Automatic versus ten dedicated workers is not equal work per
algorithm: scheduling, cooperation and per-search allowances differ. No quality,
C++ parity or default-policy superiority follows from a shorter invocation.

## First registered observations

[Run 39843491836](https://github.com/MarcosPTProenca/rko-solver/actions/runs/39843491836)
passed all registered cases and existing seed/objective/accounting/native-team
checks. The selected [compact reference](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/references/invocation-profile-reference.json)
retains manifests, host summaries, input hashes and aggregation definitions. These
are descriptive three-observation medians, not acceptance intervals.

- In the online startup probe, parent factory construction was about 4 ms;
  manager startup about 113 ms. All observed workers reached their first decode
  around 250 ms (one worker) or 305 ms (four). Internal solve elapsed was about
  500/558 ms with a 200 ms search allowance. Stages overlap: these are not additive.
  This does not justify dropping required parent-side validation for speed.
- In the offline portfolio probe, automatic selection used four workers on its VM.
  Constructor-plus-solve medians were about 1.40 s versus 1.77 s with ten dedicated
  workers; sampled coordinator/direct-child RSS-sum medians were about 317 versus
  620 MiB. Allocations/search work differ, so this is not an equal-work speed or
  quality comparison. Outer-minus-internal time was only about 38–41 microseconds,
  including constructor/return overhead, in these cases.
- In the four-worker GEMM initialization-prefix case, actual native teams were
  four per worker when inherited and one when launch-limited. Internal solve
  medians were about 1.94 s versus 0.34 s. This illustrates oversubscription risk,
  not a general RKO speedup: prefixes, warm-up and callback budgets are specific.
  The existing parent-only negative controls passed; limiting the coordinator did
  not establish a limit in its spawned workers. See [launch guidance](native-threads.md).

The VMs differed: startup Intel Xeon Platinum 8573C, portfolio AMD EPYC 7763,
callbacks AMD EPYC 9V74. Do not compare absolute times across these probes.

**Historical correctness observation:** this pre-fix portfolio reference retains
four NumPy `RuntimeWarning: invalid value encountered in divide` occurrences during
standard-deviation normalization. A deterministic two-elite/two-gene reproducer
localized the same warning to BRKGA-CS community construction. The
[defined-correlation contract](cpp-compatibility.md#brkga-cs-undefined-correlation)
now explicitly excludes undefined/numerically non-normalizable edges rather than
silencing warnings. It also removes constant-array edges fabricated by rounding,
so those trajectories can change. The historical reference/stderr is not rewritten
as warning-free data or merged with corrected-source measurements.

The hosted invocation workflow now treats numerical RuntimeWarnings as errors,
including in spawned children, and reruns for this correlation code/test. Original
seed/objective/count checks remain. Passing these checks does not prove every
algorithm invariant, C++ parity or release readiness.

## Reproduction and next decisions

Run from the repository root on noncompeting resources, or dispatch the hosted
workflow after merge. Inspect manifests/host files and raw observations together.
Do not run these probes locally while the protected historical benchmark is active.
Use a separate reference and never append to that benchmark's cache or results.

```bash
uv run --locked --with threadpoolctl==3.7.0 python -m benchmarks.runtime.startup_stages --repeats 3
uv run --locked --with threadpoolctl==3.7.0 python -m benchmarks.runtime.process_budget
uv run --locked --with threadpoolctl==3.7.0 python -m benchmarks.runtime.native_threads \
  --repeats 3 --dimension 128 --evaluations 64 --workers 1 4
```

Use the observed scopes to select one next question. Do not reopen rejected random
batching because a profiler shows a large loop, silently change process/native
thread defaults or select Rust/Cython from synthetic prefix performance. Real
expensive Python domains, full algorithm stages, adaptive trajectories, independent
quality/time-to-target and current C++ equal-resource studies remain separate work.
