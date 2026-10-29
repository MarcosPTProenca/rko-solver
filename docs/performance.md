# Performance target and native-backend evaluation

**Status: an optional native Rust engine exists; C++ parity or better is established only for
the [measured regimes](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md#native-engine-versus-c-and-python).**
The target is to **match or exceed the pinned C++ RKO on declared workloads**, in
solution quality and time under comparable resources. A language change, passing
CI, higher evaluations/second or a faster isolated kernel does not meet that target
by itself. Results outside the evaluated workload/resource envelope remain unknown.

## Outcome of the native-backend evaluation

The fixed-work profile (`benchmarks/runtime/performance_profile.py`) attributed most of the
per-evaluation time of cheap objectives to library bookkeeping (blending and RNG draws,
callback wrappers) spread over every algorithm, not to one kernel. A single optional
kernel therefore could not remove the cost, and the implemented route is the one this
page lists as a whole Rust engine: search loops, operators, per-search RNG streams, the
shared pool, Q-learning and a thread scheduler in `engines/rust/rko-core`, bound to Python
through PyO3 (`engine="rust"`, [guide](native.md)). Python callbacks still run
through the interpreter and serialize when its global lock is enabled;
block-scored NumPy objectives amortize
the call; Numba objectives and native problems run without the lock.

[The study](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md#native-engine-versus-c-and-python) measured the engine
on held-out instances against the pinned C++ reference. Two findings changed the design: a
first held-out round showed the default portfolio worse than C++ on TSP, and the cause was
the order of the local-search neighborhoods (random in the reference, a fixed cheap-first
order here), found by measuring the first descent rather than by comparing totals; and for
Python-bound problems one worker thread beat ten. The sections below keep the original
evaluation criteria, which remain the standard for any further claim.

## What the current evidence establishes

The Python implementation is Python plus NumPy, with no compiler dependency for users of a
prebuilt wheel.
Several kernels and allocation paths have measured improvements over earlier Python
versions; [each benchmark identifies its workload and limits](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md).
Those measurements do not establish whole-solver C++ performance parity.

The [1,440-pair prefill study](prefill-expanded-work.md) finds broader but not
universal IPR benefits, mixed GRASP and mostly neutral MultiStart. Its 500-variable
GRASP cases cannot finish their first construction within either evaluation cap;
all 102 wins there retain startup best. Fixed-work diagnostics and complete-cycle,
real-clock quality experiments answer different questions. We need both.

The historical long-running Knapsack comparison uses Python revision `7ee93ca`,
which predates subsequent contract and startup corrections. Preserve that run as
historical evidence; do not combine its rows with current-code results or infer the
current library's performance from it. The historical runner is now available as
`benchmarks/engines/cpp_python_knapsack.py`, with the completed two-second reference clearly
separated in [benchmark documentation](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md#python-versus-c-knapsack-comparison).
Fresh invocations execute the current checkout, not the archived implementation;
use new source-identified output/cache locations rather than resuming archived campaigns.
The integrated runner's TOML/helper fingerprint checks prevent unsafe resume across
those changes. It still needs the additional resource/stage diagnostics below to
serve as a full current-revision acceptance test. No unfinished long-run results
are published by integrating its code.

The [baseline contract checklist and fixed-work diagnostic](performance-baseline.md)
provide source-identified records and optional function-level profiling for the
initial investigation. This is a restricted serial diagnostic, not whole-solver
performance evidence or a selected native backend.

The [invocation and resource study](invocation-profile.md) reuses the startup,
portfolio and native-callback probes in fresh hosted interpreters. Its registered
scopes separate constructor-plus-solve time, internal search timing and sampled
RSS; they are diagnostic evidence, not an achieved optimization or parity claim.

## The C++ acceptance target

Reference: [RKO C++ v2, revision
`126e35873e50dc308795b58b8c75d37f8de4c857`](https://github.com/RKO-solver/RKO_Cpp_v2.0/tree/126e35873e50dc308795b58b8c75d37f8de4c857).
Use mathematically valid objective arithmetic; the historical Knapsack experiment
uses an identified 64-bit arithmetic correction rather than relying on signed
integer overflow. Record the patch hash and label the variant, not just “C++”.

Before an experiment, fix the problem families, dimensions, budgets, resource
limits, target gaps and acceptance rules. The minimum comparison record is:

| Axis | Required comparison |
| --- | --- |
| Validity | Independently decode/re-evaluate returned keys; report feasibility and every failed run |
| Quality | Best feasible gap/quality at equal end-to-end and search-only budgets, with initialization costs visible |
| Time to target | Reach rate plus censored time-to-target; do not drop failed/non-reaching trials |
| Throughput | Separate fixed-work callback/framework diagnostics; not a replacement for quality results |
| Resources | Same machine class and available CPU budget; actual workers/native teams, process CPU and memory accounting |
| Repeatability | Multiple seeds and instance families, raw rows, source/data/config/toolchain hashes and declared uncertainty |
| Coverage | Each claimed algorithm/portfolio, offline and real adaptive modes, short and long budgets, complete-stage counts |

C++ fills its shared pool before search; Python starts empty unless prefill is
opted in. The two implementations also have different uniqueness, RNG and scheduling
policies. See [compatibility](cpp-compatibility.md) and [initialization](driver-initialization.md).
Equal pool capacities or nominal `time_limit` values do not establish equal work.
Python serial time is per selected algorithm, and opt-in startup has its own
allowance. Always show complete invocation time as well as component times.

The reference [C++ driver](https://github.com/RKO-solver/RKO_Cpp_v2.0/blob/126e35873e50dc308795b58b8c75d37f8de4c857/Program/src/Main/main.cpp)
requests one OpenMP thread per selected metaheuristic. Python's process workers
are CPU-bounded by default. Comparing their defaults is a product-default study,
not an equal-resource engine test. Verify actual concurrency and queued search
budgets; do not assume setting an environment variable alone equalizes execution.

Do not claim parity from a nonsignificant difference or a pooled average that
hides regressions. Use predeclared non-inferiority/superiority rules and uncertainty
per claimed regime; if the evidence cannot exclude meaningful regression, report
**inconclusive**. No universal “at least C++ speed” guarantee is justified for
arbitrary Python callbacks, dimensions and hardware.

## Why a Python API with Rust underneath is feasible, but not sufficient

PyO3 supports native Python extensions and Maturin supports mixed Python/Rust
packages [1, 4]. The public `RKO`/result interface can remain Python-facing.
However, the current evaluation path is:

```text
search/local-search candidate
    -> deadline/cancellation check
    -> detached chromosome snapshot + decoder-protecting copy
    -> user Python decoder
    -> deadline/cancellation check
    -> user Python objective + finite-value validation
    -> completed-evaluation count and best-incumbent retention
```

These protections are real behavior in
[`SearchSession`](https://github.com/MarcosPTProenca/rko-solver/blob/main/src/rko_solver/orchestration/search.py), not disposable overhead.
Moving the search loop into Rust still calls Python for arbitrary existing
callbacks. On GIL-enabled CPython those calls require interpreter attachment/GIL
access. Rust-only work can detach; ordinary Python callbacks cannot execute while
detached [1, 2]. NumPy/native callbacks may themselves release the GIL, so the
actual callback mix must be measured rather than assumed.

A useful bound for a **kernel-only** change is `S = 1 / (u + (1-u)/s)`, where `u`
is the measured unchanged fraction and `s` is that kernel's speedup. If unchanged
callback/other work is 80% of runtime, even an infinitely fast remaining core is
limited to 1.25x. This is an illustrative model, **not a measured RKO fraction**.
A full-engine change that also removes IPC/startup has a different decomposition.

## Options worth comparing

| Option | Potential benefit | Main limitation / adoption condition |
| --- | --- | --- |
| Existing Python + NumPy | Lowest packaging cost; optimize proven bulk/allocation paths | Python control and callback overhead remain; NumPy arithmetic is already native |
| Optional coarse Rust kernels through PyO3 | Keep Python API/fallback; move substantial callback-free work across one boundary | Per-gene/per-evaluation FFI may erase gains; profile and prove whole-search benefit |
| Full Rust search engine, Python callbacks | Native state, candidate generation and potentially less manager/process overhead | Repeated Python callbacks, new scheduler/RNG/safety contracts and much larger validation cost |
| Native engine plus explicitly native problem adapter | Can keep most work outside the interpreter; strongest route toward compiled end-to-end execution | Extra adapter work for users; native-only wins are not general Python-callback gains |
| Bind/refactor the C++ implementation or use another compiled extension | A comparison alternative; Rust is not inherently faster than C++ | Existing application/global-state integration, compatibility, ownership and build work remain |

**Historical discovery criteria:** the following proposal predates the implemented
engine selector. It is not a second implementation plan or a description of today's
packaging. Profile first;
then a bounded optional-kernel experiment if callback-free work can justify it.
Evaluate a whole Rust engine only if larger native ownership is necessary to meet
the C++ target. A compiled adapter must remain optional for a general-purpose
library, not a hidden Knapsack-specific repair. No engine-selector API is designed
or introduced here (the baseline predates `engine`; `execution` means process, thread
or sequential Python scheduling only).

Static source inspection suggests investigating repeated neighborhood/GRASP/IPR
control, candidate construction, blending/RNG bookkeeping, callback boundaries and
manager RPCs. This is a **candidate list, not a measured bottleneck ranking**.
Existing vectorized PSO/NumPy blocks may offer less headroom than scalar control.
Kernel, Python-callback and native-callback paths must be measured separately.

## Gates before adopting a native component

1. **Profile the real cost.** Separate reader/import/spawn, initialization,
   candidate generation, copies, decoder/objective, local search, elite IPC and
   adaptive control. Use cheap, expensive/native and application callbacks, plus
   algorithm-stage completion. Do not compete for resources with an active protected campaign.
2. **Use a disposable, bounded experiment.** One kernel or one algorithm, same
   objective/problem and work limits, Python fallback, release-mode native build.
   Include native compute with and without Python callbacks to expose bridge cost.
   Reject a microbenchmark-only win or a material regression in a claimed regime.
3. **Preserve contracts.** Callback payload/mutation isolation, retained arrays,
   errors/nonfinite rejection, incumbent recovery, counts and cancellation checks
   stay intact. Random draw order, first-minimum ties, signed zero, partial blocks
   and extreme GRASP lattice bounds need differential tests or explicit deviations.
   Do not silently batch/reorder stateful callbacks or change adaptive timing.
4. **Own native memory safely.** NumPy borrowing checks do not prevent arbitrary
   Python/native code from mutating aliased storage [3]. Do not reuse borrowed
   scratch after giving a callback a retainable/mutable view, or promise zero-copy
   merely because Rust compiles. Test noncontiguous arrays and concurrent access.
5. **Prove delivery.** The default Python search engine must remain usable.
   Source installation builds the native extension and requires Rust; a matching
   prebuilt artifact needs no compiler. Test built distributions, packaged parameters,
   exceptions and results on supported
   platforms. Native wheels add OS/architecture/interpreter/NumPy and toolchain
   compatibility work; Linux distribution needs compatible manylinux/musllinux
   handling [4, 5]. The implemented mixed Python/Rust build uses Maturin; current
   configuration and pinned native dependencies live in `pyproject.toml` and
   `Cargo.lock`. The original discovery did not select those later implementation details.
6. **Separate ABI claims.** Stable-ABI and GIL/free-threaded wheel compatibility
   are distinct, version-sensitive concerns [6]. Do not assume a single `abi3`
   wheel covers CPython 3.14's free-threaded build or every NumPy binding. Current
   release checks install the source distribution on free-threaded CPython 3.14 on Linux
   and run the installed-package smoke without enabling the global interpreter lock.
   Published wheels remain regular `cp314-abi3` builds; this source-install check does not
   certify free-threaded wheels or prove a threading speedup.
7. **Resolve attribution and release policy.** The upstream C++ MIT license is
   visible [7], but it does not settle the combined project's imported-code review
   in [NOTICE](https://github.com/MarcosPTProenca/rko-solver/blob/main/NOTICE). The optional C++ engine
   vendors the pinned headers unchanged with that license and attribution; the owner's
   confirmation of the combined licensing remains a release prerequisite.
8. **Return to the actual C++ target.** Run fair, current-revision multi-instance
   comparisons and publish losses/failures as well as wins. Only then reconsider
   default engine selection or performance claims.

## Primary sources

External documentation reviewed 2026-10-22. Upstream `main` pages evolve; examples
and versions must be revalidated when an experiment is implemented.

1. [PyO3 performance: callbacks, interpreter attachment and native work](https://pyo3.rs/main/performance).
2. [PyO3 parallelism](https://pyo3.rs/main/parallelism).
3. [rust-numpy borrowing rationale and limitations](https://pyo3.github.io/rust-numpy/numpy/borrow/index.html).
4. [Maturin mixed-project layout](https://www.maturin.rs/project_layout).
5. [Maturin distribution and portable wheels](https://www.maturin.rs/distribution.html).
6. [PyO3 building/distribution and stable ABIs](https://pyo3.rs/main/building-and-distribution).
7. [Pinned C++ MIT license](https://github.com/RKO-solver/RKO_Cpp_v2.0/blob/126e35873e50dc308795b58b8c75d37f8de4c857/LICENSE).
