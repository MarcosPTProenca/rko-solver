# Full-driver initialization: Python and the pinned C++ reference

## Scope and outcome

The default remains unchanged after this source-contract audit; this is not a
quality/timing comparison. Python's shared-pool capacity alone does **not** request a
prefilled population. An explicit opt-in reference-style phase is described below. The pinned C++ driver creates a full shared pool before
launching algorithms. Matching `pool_size=10` to `sizePool 10` therefore does not
match initialization work or starting search context.

Reference: `RKO_Cpp_v2.0`, revision
`126e35873e50dc308795b58b8c75d37f8de4c857`, `Program/src/Main/main.cpp`,
`Program/src/Method.h`, and `Program/src/MH/IPR.h`.

## C++ startup pool

`Main/main.cpp` starts its run clock, clears/resizes the shared pool to `sizePool`,
then calls `Method.h::CreatePoolSolutions` before entering the OpenMP algorithm
loop. For positive capacity P and chromosome length n, that function:

1. Generates P random-key vectors and calls `Decoder` on each.
2. Sorts all entries by increasing objective using selection sort.
3. Scans backward once. Whenever adjacent entries have equal objective values,
   it performs the integer loop `j < 0.2*n`: draw a coordinate and replace its key,
   then re-evaluate that entry once. Coordinates can be drawn repeatedly.
4. Re-sorts if at least one entry was perturbed.

This is a **single cost-tie perturbation pass**, not a chromosome-uniqueness
retry loop. For finite scores, let T be the number of adjacent equal-cost pairs
in the initial sorted pool. The source implies P+T decoder calls, between P and
2P-1 inclusive; a flat objective gives 2P-1 calls. These are source-derived
bounds, not measured application counts. Equal costs and duplicate chromosomes
can remain. The stock C++ insertion policy also rejects repeated objective costs;
Python instead retains [one best observation per chromosome](elite-diversity.md).
Copying these policies blindly would undo an explicit Python diversity contract.

The driver records `pool[0]` as its initial best. If a run restarts, it can call
`CreatePoolSolutions` again; that is additional work, not a one-time universal
constant. Individual algorithms may also create their own private populations.
The shared startup pool is not a substitute for those private populations.

## Default Python startup

`RKO._execute` in `src/rko_solver/solver.py` creates an empty `SolutionPool` and constructs
selected algorithms without driver-level candidate evaluation. Sequential execution
shares that pool across algorithms in order; threads share it concurrently.
Later sequential searches can therefore inherit earlier elites, and thread startup
observations depend on peer progress. There is no global prefill barrier.

Process execution creates an empty managed pool in
`src/rko_solver/orchestration/processes.py::run_processes`. Worker constructors report
ready before their searches run. Releasing the workers starts search work, not
a shared pool-initialization phase. A queued search can inherit elites from
previous/parallel searches; no worker is promised a full initial pool.

Candidates generated inside algorithms pass through `SearchSession`: completed
finite objective calls contribute to that search's count and incumbent, including
interrupted private-population initialization. Current standalone IPR inserts
its first observation when needed and tries at most one extra random candidate
to obtain a pair. It does **not** fill the configured capacity or retry until
all chromosomes are unique. See [the IPR initialization distinction](ipr-diagnosis.md).

## Clock and budget distinctions

C++ `Decoder` returns an objective value; Python separates decoding from objective
evaluation and counts completed finite objective calls. Compare equivalent
candidate-evaluation work, not the sum of Python decoder and objective callbacks.

| Boundary | Pinned C++ driver | Current Python |
| --- | --- | --- |
| Reported elapsed origin | Before initial shared-pool creation | Public solve includes configuration/startup |
| Shared startup callbacks | Decoder calls before algorithms start | No driver-level seed callbacks |
| Algorithm search clock | Fresh local clock, e.g. IPR uses `MAXTIME*restart` | Sequential/thread: per-search duration; process: remaining common-deadline allocation |
| Initialization stop boundary | No time/cancellation check inside `CreatePoolSolutions` | Ordinary algorithm callbacks use the session's cooperative boundary |
| Evaluation accounting | No Python-style per-search initialization report in the reference driver | Completed callbacks counted in each search report |

A clock that starts before pool creation does **not** prove its cost was deducted
from algorithm search time. C++ initializes `end_time` before pool creation and
does not refresh it before the first outer-loop condition; algorithms then start
their own clocks. Do not describe this as a strict common deadline. Python also
allows a first valid evaluation after startup has consumed its callback deadline;
active callbacks cannot be interrupted. Neither policy guarantees a hard wall cap.

## Protected benchmark interpretation

The separate `cpp-python-knapsack-benchmark` worktree is pinned at `7ee93ca`.
Its harness sets C++ `sizePool 10`, Python `pool_size=10`, and `restart 1`; it does
not replace C++ pool creation with an equivalent Python prefix. Its C++ telemetry
clock is inserted at the run-clock boundary, before pool creation. Historical
results must retain those semantics; current IPR bootstrap/diversity corrections
must not be retroactively attributed to that pinned Python revision.

This audit neither edits its source/cache/CSV nor restarts or duplicates that run.
It does not establish that startup differences alone caused any feasibility or
quality gap. The same nominal capacity and duration are insufficient evidence of
matched initialization, callback work or search time.

## Opt-in reference-style startup

Enable `RKO(..., prefill_pool=True)` or `[run] prefill_pool = true`. The packaged
default is false, old configurations may omit it, and explicit Python booleans
override TOML. The CLI supports `--prefill-pool` and `--no-prefill-pool`; absent
flags follow TOML. Legacy `run(..., prefill_pool=True)` keeps its tuple return.

The phase creates P private slots, uses the reference selection-sort swap order,
then performs its single backward cost-tie perturbation pass and re-sorts. Working
slots are kept separate from elite admission so worse mutated replacements are not
silently rejected before completing the reference pass. Completed final/partial
slots and the phase's best observation are published through ordinary Python
chromosome-unique admission. There are no retries to fill duplicate slots; admitted
occupancy may be below capacity. Full phases use P..2P-1 objective observations.
Slot storage is O(P*n), selection sorting O(P²); no resource improvement is claimed.

Startup uses an independent owned stream derived from a reserved grandchild seed
branch `(0, 0)`, independent of portfolio length. Existing algorithm seed derivation
is unchanged. Python/C++ random engines and final admission policies still differ;
this is not exact seeded trajectory parity.

Startup has **its own cooperative allowance equal to `time_limit`**. This is a
safety extension beyond the reference's unchecked fill. A first finite observation
is allowed; on expiry completed partial context and the best observation survive.
Active callbacks cannot be interrupted. The selected searches then receive their
usual fresh allowances: startup is not deducted from their budgets. Total elapsed
can grow by roughly another requested duration, plus normal setup and callback
overruns. A `time_limit` startup report can also describe an overrun after all
planned observations completed; it does not guarantee a partial pool.

Sequential/thread startup runs in the caller after constructor validation and before
search execution. Process startup uses the first existing worker after all workers
are ready; no callback runs in the parent/manager and no extra search worker is
added. Other workers wait until the initialization report, then the parent starts
a fresh common search deadline and releases searches. If that search window expires
before a worker can execute, opted-in runs retain the already evaluated shared
incumbent and report the search as not started, without another callback. Default-off
runs retain their preceding error behavior when no worker search could execute.
The initialization watchdog
uses its own allowance plus the existing five-second process grace; a blocked
callback or failed initializer tears down waiting workers. Decoder/objective or
nonfinite failures fail the solve rather than masquerading as successful partial
startup.

`result.initialization` is an immutable `PoolInitializationMetadata` report with
actual `random_state`, `n_evaluations`, `elapsed_seconds`, admitted `pool_size`,
`stop_reason` and `time_limit`; it is None when disabled. `result.searches` remains
per-algorithm only. `result.n_evaluations` includes startup plus searches exactly
once. No objective is re-run to publish the best or calculate metadata.

Callback-owned mutable/global/random state is not cloned or reset. Additional
startup observations can affect subsequent user callbacks, including the designated
process worker's own callback state; other workers have their separate process
copies. Library RNG isolation does not promise stochastic-backend equivalence.

## Remaining evidence

Compare equal initial vectors and total completed-work budgets against the default
empty-pool policy across general numerical/permutation objectives and relevant
instance families. Startup observations must be included, not compared only through
per-search reports. Separate quality from callback throughput before selecting
native kernels. A [small matched-work pilot](prefill-matched-work.md) found
consistent IPR improvements but mixed GRASP effects at one offline dimension/budget;
keep prefill opt-in. The [hosted expansion](prefill-expanded-work.md) confirms
broader but non-universal IPR benefits and identifies incomplete GRASP stages at
small budgets, including startup-only wins. General quality, feasibility,
throughput, memory and C++ parity remain unestablished.

Default-off regression probes use the real public sequential/thread dispatchers,
factory, callback sessions and result metadata, replacing only one algorithm body.
Capacities 1, 10 and 100 all enter construction/search with zero entries/evaluations;
one completed observation is reported once. They characterize driver policy,
not every algorithm's private initialization. Separate opt-in tests exercise real
sequential/thread/process startup, callback ownership, grouped worker caps, completed
totals, partial timeouts, nonfinite/errors, blocked-worker cleanup, configuration
precedence and CLI forwarding. The C++ observations remain source-audited, not new
compiled-C++ application or quality measurements.
