# Usage and API guide

[Start with the quickstart](getting-started.md) for a first sequential solve. This guide
covers configuration, execution modes, callbacks, result accounting and stopping.
For one algorithm use a [public solver such as `SA(...)`](standalone.md), followed
by the same `.solve(time_limit=...)` call and settings as RKO.
The [parameter reference](parameters.md) explains every fixed scalar and adaptive
mode; the [extension guide](extending.md) covers custom operators and algorithms.
Commands and example paths assume the repository root. The Python library needs
neither a C++ checkout nor a Rust compiler.

- [Solver configuration](#class-based-solver-api)
- [Execution modes](#execution-modes)
- [Offline tuning](#tuning-offline-parameters-from-python)
- [Process-safe callbacks](#spawn-compatible-callbacks)
- [Results and accounting](#result-accounting-and-configuration-snapshots)
- [Custom problems and stopping](#adapting-a-custom-problem)
- [Terminal adapter](#run-an-instance-from-the-terminal)

Contributor setup and validation are also summarized in [CONTRIBUTING](https://github.com/MarcosPTProenca/rko-solver/blob/main/CONTRIBUTING.md).
C++ comparison goals and native-engine limits are in [performance](performance.md).

## Development setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/MarcosPTProenca/rko-solver.git
cd rko-solver
uv sync --locked --group dev
uv run --locked pre-commit run --all-files
```

For automatic commit checks, run `uv run pre-commit install` if Git uses its
standard hooks directory. If `core.hooksPath` is centrally managed, integrate
pre-commit with those existing hooks instead; no global Git settings are changed
by this setup.

The distribution is named `rko-solver` and imports as `rko_solver`; version `0.3.0a2`
identifies this prerelease, not a PyPI publication. Attribution and licensing
boundaries are in [NOTICE](https://github.com/MarcosPTProenca/rko-solver/blob/main/NOTICE).

## Class-based solver API

`RKO(decoder, cost, n_variables, ...).solve(time_limit)` minimizes
`cost(decoder(keys))`. The first argument may instead be a problem object that carries
the problem definition: `rko_solver.Callbacks(decoder, cost, n_variables)`,
`rko_solver.Numba(function, n_variables)`, `rko_solver.Blocks(function, n_variables, batch_size)` or a
built-in such as `rko_solver.problems.Knapsack(...)`; then `cost` and `n_variables` are not passed. The constructor stores configuration without reading files
or starting workers. Each solve builds a fresh portfolio and returns an immutable
`RKOResult` with `keys`, `cost`, `elapsed_seconds`, `engine`, `execution`, `algorithms` and
`random_state`. `solver.result_` holds the latest successful result and is cleared
before every attempt, including failed solves.

### Constructor options

The first three arguments are required in the short form. The others are keyword-only:

| Argument | Meaning / default |
| --- | --- |
| `decoder` | Callable receiving a detached NumPy key vector and returning your domain payload |
| `cost` | Callable mapping that payload to a finite scalar to minimize |
| `n_variables` | Positive integer chromosome length |
| `algorithms=None` | Nonempty selection of canonical names, otherwise packaged TOML portfolio |
| `adaptation=None` | Configured mode; `"fixed"` settings or `"adaptive"` Q-learning where supported |
| `control=None` | Legacy alias: `0` fixed or `1` adaptive; conflicting aliases are rejected |
| `pool_size=None` | Configured elite capacity; not a prebuilt population |
| `prefill_pool=None` | Configured startup policy; packaged default is off |
| `params_dir=None` | Custom TOML directory, otherwise installed resources |
| `engine="python"` | `"python"`, `"rust"` (compiled engine), `"cpp"` (reference engine, opt-in build) or `"auto"`; see [Rust and C++ engines](native.md) |
| `execution=None` | Python engine only: `"process"` (default), `"thread"` or `"sequential"`; rejected for other engines |
| `n_jobs=None` | Process or Rust worker cap; process default uses discovered CPU capacity, Rust default one thread per algorithm |
| `random_state=None` | Nonnegative library seed, or fresh assigned seeds |
| `verbose=False` | Opt-in lifecycle/shared-pool progress to stdout |
| `max_evaluations=None` | Positive number of completed evaluations after which **each** algorithm stops, in every engine; see [evaluation limit](#evaluation-limit-and-reproducibility) |
| `history_limit=0` | Opt-in per-search scalar history cap, 2–10000; see [interactive reports](reporting.md) |
| `trace=None` | Opt-in bounded original-decoder events with `TraceConfig`; see [observed decisions](decision-traces.md) |
| `hyperparameters=None` | Partial named scalar overrides, requiring `control=0` |

`solve(time_limit)` requires positive finite seconds. Invalid dimensions/settings,
callback serialization and nonfinite costs are rejected, not silently repaired.
Process worker failures can surface as `RuntimeError` with their original cause;
startup/execution watchdogs can raise `TimeoutError`. Parameter I/O can raise
`OSError`; user callback exceptions are not converted into a successful result.
A fresh solve validates again, so mutating settings can make a later attempt fail.
Do not call `solve` concurrently on the same instance. See stopping details below;
these are not guarantees that arbitrary callbacks terminate promptly.

Select `algorithms`, `control` and `pool_size` directly in Python; omitted values
use the packaged TOML defaults. By default `pool_size` is capacity, not an initial
population. Opt in with `prefill_pool=True` (CLI `--prefill-pool`) for
[reference-style shared initialization](driver-initialization.md#opt-in-reference-style-startup).
Startup has its own cooperative allowance equal to `time_limit`, followed by fresh
search allowances; total elapsed can increase. `result.initialization` reports its
work separately, and `result.n_evaluations` includes startup plus searches once.
The [1,440-pair quality expansion](prefill-expanded-work.md) finds broader but
non-universal IPR benefits, mixed GRASP and mostly neutral MultiStart; keep startup
opt-in. Large-GRASP gains can be startup-only before construction completes.

SA now implements its requested online control
rather than falling back to offline parameters; use `control=0` for its legacy
fixed-parameter path. See [adaptive SA and migration](sa-control.md).
GRASP now samples the [wider reference lattice](grasp-line-search.md) in both
control modes; seeded trajectories and callback counts change without a new option.
Its grid step is halved only after a construction that fails to improve the global
best, and the initial random vector is scored and published without refinement,
matching the reference grid progression; seeded trajectories also change here.
The [alpha axis follows the reference compatibility policy](grasp-alpha.md):
it affects adaptive states/actions, not construction directly; no RCL is implied.
With `control=0`, `hyperparameters` supplies partial
per-algorithm scalar overrides without editing files. `params_dir` remains
available for custom defaults and adaptive grids. Library calls are quiet by default; `verbose=True` on `RKO` prints lifecycle
and attributed shared-pool improvements. `rko_solver.run(...)` retains the old tuple API
and also accepts `verbose=True`.

## Execution modes

The `engine` chooses who runs the search. `execution` only exists for the Python engine:

- **`engine="python", execution="process"` (default):** `n_jobs=None` uses at most
  `min(number_of_algorithms, os.process_cpu_count() or 1)` search workers,
  plus one manager hosting the shared elite pool. CPU discovery is affinity-aware;
  unavailable discovery falls back to one. A positive `n_jobs` overrides this
  automatic cap, even above available CPUs, without dropping algorithms:
  assignments are round-robin and each worker runs its
  assigned searches sequentially. Before each search, that worker divides the
  time left before the common deadline by its number of remaining searches.
  For example, ten algorithms with `n_jobs=4` form groups of 3, 3, 2 and 2;
  their initial shares are approximately one third or one half of the budget.
  Early completion leaves more time for queued searches in the same worker;
  no search extends the original deadline. This replaces the previous fixed
  longest-group slice; shorter groups no longer deliberately finish early.
  Smaller limits change cooperation and per-algorithm budgets; they are not a
  promise of better optimization quality. Active callbacks can overrun a slice
  and leave later searches less time or no time. Algorithms sharing a worker
  also share its callback instances; mutable callback state is not reset between
  searches. Native BLAS/OpenMP threads are separate from this worker limit;
  use the [native-thread launch guide](native-threads.md) for numerical callbacks.
- **`engine="python", execution="thread"`:** legacy shared-memory execution; arbitrary
  local callbacks work, but CPU-bound Python code is subject to the standard GIL.
- **`engine="python", execution="sequential"`:** debugging/reference execution with shared
  elites. Each algorithm gets its own time budget, so total search time can approach
  algorithm count times `limit_time`. This is not an equal-wall-time benchmark
  against concurrent modes.
- **`engine="rust"`:** the compiled engine; Python callbacks serialize on the
  interpreter lock, built-in problems do not. See [Rust and C++ engines](native.md).
- **`engine="cpp"`:** the C++ reference algorithms, available in builds made with
  `RKO_BUILD_CPP=1`.
- **`engine="auto"`:** short probes choose among the others for your problem.

The deprecated `backend=` argument (`"process"`, `"thread"`, `"serial"`, `"native"`,
`"cpp"`, `"auto"`) and `RKO.native`, `RKO.compiled` and `RKO.batched` still work, with a
`DeprecationWarning`, and will be removed in 0.4.

IPR consumes elite pairs. If fewer than two elites exist at startup, it inserts
its initial evaluated candidate and tries at most one additional random seed,
using the same decoder/objective and assigned deadline. This makes standalone or
first-in-group IPR able to relink without waiting for a producer queued behind it.
An already usable pool retains its initialization draw order. Pair occupancy alone
is not population diversity for custom providers. Stock pools now retain
[one best observation per chromosome](elite-diversity.md), including managed
pools; IPR skips identical effective parents. See the
[fixed-work diagnosis](ipr-diagnosis.md) for limits. IPR now
[omits unchanged-block evaluations](ipr-block-pruning.md), retaining virtual
path decisions for deterministic scoring; stochastic/stateful callbacks receive
fewer observations and can change results. Use a pool capacity
of at least two; a pool unable to retain a pair still uses bounded waiting after
those seed attempts. Seeds are not domain-specific feasibility repairs. Process
occupancy queries transfer only a count rather than the entire pool.

Algorithm defaults live in packaged `rko/parameters/` TOMLs.
Use `algorithms=["SA", "GA"]` to select a smaller portfolio without copying files.
Custom default TOMLs can still be supplied through `params_dir`.
Constructors and configured runs validate scalar domains, population sizes and
joint constraints before execution. Online grids retain named axes and exclude
infeasible joint combinations (such as reversed perturbation bounds); an entirely
infeasible grid is rejected. No scalar value is silently truncated.

### Tuning offline parameters from Python

Use the existing offline parameter names, grouped by selected algorithm:

```python
from rko_solver import RKO

solver = RKO(
    decode, objective, n_variables=64,
    algorithms=["PSO", "GA"], adaptation="fixed",
    hyperparameters={
        "PSO": {"p_size": 12, "w": 0.6},
        "GA": {"size_pop": 20, "prob_mut": 0.1},
    },
    n_jobs=2, random_state=42,
)
```

This uses the importable callbacks and guarded process entry point shown below.
Each `solve` snapshots the mappings, resolves aliases such as `GeneticAlgorithm`
to `GA`, and merges the values into fresh TOML defaults. Unspecified values keep
their defaults; neither input mappings nor installed files are modified. Required
TOMLs must still exist, including when using a custom `params_dir`.

Unknown parameter names, duplicate alias mappings, unselected algorithms,
nonfinite values, booleans and invalid scalar/joint domains raise `ValueError`
before execution. Exact integral floats are accepted for integer parameters;
fractional values are never truncated. Unknown-key errors list supported scalar keys.
All engines and execution modes, including grouped process workers, receive the same owned settings.

Nonempty scalar overrides require **`control=0`**. Adaptive grids remain configured
through named `[online]` TOML axes; `control=1` rejects scalar overrides rather than
silently replacing tuned values during adaptation. Result reports retain the
validated initial scalar settings and actual adaptive axes used by each search.

Low-level kernel constructors (explicit implementation-module imports, not the
managed public solver constructors) use package-provided local search and shaking
by default. Kernels that perform local search accept a typed
`local_search_operator`; SA, ILS and VNS also accept `shaking_operator`. These
callables are optional overrides, not required problem callbacks. Their public
contracts are `LocalSearchOperator` and `ShakingOperator` from
`rko_solver.metaheuristics`; local-search overrides receive `pool=` as a keyword.

`adaptation` (or legacy `control`), `pool_size` and `prefill_pool` are supported under `[run]`. Remove the formerly
ignored `strategy`, `max_runs` and `debug` fields from custom configurations;
unsupported keys now raise an error. Portfolio local search currently uses
best-improvement defaults. Repeat `solve` for independent trials and use standard
Python logging for diagnostics.

### Spawn-compatible callbacks

Use module-level functions, `functools.partial`, or callable instances carrying
serializable state. Lambdas and nested functions are rejected in process mode.
The defining module must be importable in a fresh interpreter. Run from a script
rather than an interactive notebook cell, and guard the entry point:

```python
import numpy as np
from rko_solver import RKO


def decode(keys):
    return keys


def objective(solution):
    return float(np.sum(solution**2))


if __name__ == "__main__":
    solver = RKO(
        decode, objective, n_variables=10,
        algorithms=["SA", "GA"], n_jobs=2, random_state=42,
    )
    result = solver.solve(time_limit=5.0)
    print(result.keys, result.cost, result.elapsed_seconds)
```

### Result accounting and configuration snapshots

`RKOResult` retains its original fields and adds immutable `searches` reports in
portfolio order, plus `n_variables`, requested `time_limit`, `control`, `pool_size`,
`n_jobs`, process-only `available_cpus` and runtime `versions` (RKO distribution,
Python, NumPy). CPU capacity is snapshotted once per process solve, so changing
affinity later does not rewrite earlier resource accounting. The derived
`n_workers` excludes pool managers and native numeric threads; it is configured
search concurrency, not measured CPU utilization.

<!-- doc-test: skip prints the result of the complete program above, which runs as its own script -->
```python
print(result.n_evaluations, result.stop_reason, result.n_workers)
for search in result.searches:
    print(search.algorithm, search.n_evaluations, search.stop_reason)
    print(search.random_state, dict(search.parameters), search.parameter_grid)
```

With `history_limit > 0`, each search also exposes immutable `.history` points
(local elapsed seconds, evaluations, incumbent cost), `.history_limit` and
`.history_observations`. Recording is bounded; compacted curves are approximate.
[Interactive reports](reporting.md) explains offline HTML/JSON exports and their
privacy and interpretation limits. Capture is disabled by default.

`n_evaluations` counts **successfully completed finite objective callback calls**
across every algorithm, including initialization and local search—not just
improvements or the winning worker. Decoder calls, rejected/nonfinite objectives,
callback-internal work and post-solve user evaluations are not counted. Failed
solves still raise and clear `result_`; they do not publish partial success reports.

Each exported `SearchMetadata` records its actual assigned seed (even when the
requested seed was `None`), effective control, validated initial scalar parameters,
ordered adaptive grid axes, assigned search seconds and measured session seconds.
Adaptive grids follow the algorithm's schema order; these are input snapshots,
not final adaptive parameters or a history of visited states. Unsupported adaptive
algorithms report effective control zero. Workers report their own constructed
settings rather than rereading files later in the parent. Reports contain no
callbacks, populations or mutable RNG objects and support standard dataclass/
pickle serialization. Aggregate properties are derived, not `dataclasses.asdict`
fields. Legacy manually constructed six-argument results have unknown accounting
(`n_evaluations=None`, `stop_reason="unknown"`). `run()` and CLI CSV formats remain
unchanged.

Search reasons are `completed` (returned before its deadline, **not** a convergence
claim), `time_limit` (deadline stopping or deadline elapsed by return), `max_evaluations`
(the [evaluation limit](#evaluation-limit-and-reproducibility) was reached), `cancelled`
(peer-failure signal), and `not_started` (queued search had no remaining budget;
its assigned `time_limit` is zero). Process search limits are actual allocations
computed at dispatch, not equal fixed fractions or the requested portfolio budget.
The portfolio reason is `time_limit` if any search used its budget or could not
start, otherwise `max_evaluations` if any search reached its limit; detailed reports
preserve the distinction. Unstarted entries retain their
nominal slice but have zero evaluations and zero elapsed search time. Cancelled peers normally belong
to a failed portfolio, which raises instead of returning a result. Session elapsed
time excludes startup and queueing; `result.elapsed_seconds` remains end-to-end.

These numeric settings and runtime versions are useful provenance, not a complete
replay artifact: callback/data identities, development commit hashes, callback RNG
state, adaptive trajectories and cooperative scheduling are not captured. Owned
seeds do not promise identical timed portfolio results. Callback overruns can still
exceed a reported budget.

### Evaluation limit and reproducibility

`max_evaluations=N` stops **each** algorithm after exactly `N` completed objective
evaluations, on the Python, Rust and C++ engines alike; `time_limit` still applies and
whichever comes first ends the search (`search.stop_reason` says which). It does not count
the opt-in pool prefill, which is reported separately, and the C++ reference's own
mandatory 10-vector prefill is likewise reported only under `result.initialization`.
Because it does not depend on machine speed, it is the way to compare engines, write
regression tests or repeat a run.

```python
result = RKO(decode, cost, n_variables, execution="sequential", max_evaluations=5000,
             random_state=42).solve(time_limit=60.0)
assert {s.n_evaluations for s in result.searches} == {5000}
```

Measured on a 60-item Knapsack with the default portfolio and a generous time limit, every
search ended at exactly the limit in all five configurations (Python sequential, thread and
process, Rust and C++), with block objectives truncated at the limit instead of overshooting
it. Two runs with the same `random_state` returned identical keys, costs and counters for
Python **sequential**, Python **process with `n_jobs=1`** and Rust with one worker (the default
for Python callbacks and block objectives); they differed for Python **thread**, for several
concurrent Rust threads and for **C++**, where algorithms exchange elites concurrently or
(C++) draw part of the descent from an unseeded generator. The limit is exact for every
configuration; only reproducibility depends on the worker setup.

### Adapting a custom problem

Keep the problem-specific reader outside RKO. Load the data once, then pass a
one-dimensional NumPy key vector to a decoder that returns your domain solution;
the objective receives that solution and must return a finite scalar to
**minimize**. For `execution="process"`, callbacks must be importable and
pickleable: use module-level functions or callable objects that hold the loaded
data. The `Spawn-compatible callbacks` example above shows a runnable setup; for
a structured problem, replace its identity decoder and sphere objective with
your own adapter. RKO itself does not parse problem files or assume a solution
representation.

`random_state` gives distinct, repeatable seed allocation to privately owned
Python and NumPy RNGs in **all three execution modes**, including threads. Library
operations do not reseed or restore caller-global generators, so independent
solver instances can overlap without contaminating each other's library draws.
Timed searches still depend on scheduling and elite exchange; identical seeds
do not guarantee identical results.

Custom decoders, objectives and operator overrides must manage their own
randomness and concurrent mutable state. In particular, `random_state` no longer
implicitly seeds global RNG calls inside callbacks, and callback-owned global
draws are not rolled back after a sequential solve. Use explicitly owned generators
for stochastic callbacks. Standalone built-in operators called outside RKO keep
their existing global-RNG behavior. A single mutable RKO instance is not intended
for simultaneous `solve` calls.

Process startup is excluded from the requested search duration. On WSL,
Windows-mounted virtual environments and editable source can dominate short runs;
see [tested Linux-local setup guidance](running.md#wsl-startup-and-virtual-environment-placement).
Workers start searching together after construction. A per-search callback session carries the
same deadline through initialization and every nested operator, stopping between
decoder/objective calls and preserving the best completed evaluation. A first
valid evaluation is allowed even if initialization used the budget. User callbacks
receive ordinary decoded payloads; candidate arrays are isolated from decoder
mutation. Low-level operator calls made outside RKO do not acquire this budget.

Startup has a 60-second watchdog; process execution has five seconds of grace
for callbacks that are still running. A blocked process callback or excessive
overrun raises `TimeoutError` and stops workers. Thread mode propagates worker
failures and cooperatively cancels peers, but cannot force-kill a blocked Python
thread. This is not a hard real-time API. Algorithm failures identify the search
and preserve exception causes. The parent cleans up processes and the manager
on failure or keyboard interruption.

The managed pool is a correctness-first baseline. IPC and process startup can
outweigh parallelism on tiny problems; no universal speedup is promised.

## Run an instance from the terminal

Like C++ `runTest INSTANCE T`, the supplied problem adapter runs Knapsack:

```bash
uv sync --locked
uv run --locked rko-solver src/rko_solver/examples/instances/kp50.txt \
  --limit-time 10 --execution process --random-state 42 \
  --output-csv results/kp50.csv
```

The terminal command shows clock/elapsed timestamps and new-best algorithms, plus
pool status approximately every second even during plateaus. Add `--quiet` for
final results only.

Pass `--params-dir configs/my-run` to select your own `config.toml` and algorithm
TOMLs. See [running an instance](running.md) for copying/editing configuration,
a four-worker portfolio, instance format, PowerShell and legacy batch selection.
The supplied command does not infer arbitrary problem formats: other problems
require their own reader/decoder, analogous to adapting C++ `Problem.h`.

The two bundled references live in `src/rko_solver/examples/instances/`, with upstream
revision and MIT license. No C++ checkout or `PYTHONPATH` setting is required.
Omit `--output-csv` to save under the caller's `results/` directory.

The decoder uses `key > 0.5`. Cost is negative profit with a penalty of 100000
per excess unit of weight, matching the reference example. Independently
computed exact optima are **31** for `kp10.txt` and **149** for `kp50.txt`. The
CSV records feasibility, objective, engine, execution, requested worker budget, seed and
end-to-end elapsed time. A blank budget requests automatic affinity-aware process
workers; it is not a measured worker count. To retain dedicated workers, explicitly
request the selected portfolio size. See [default-worker migration](process-workers.md).
These two small instances are correctness checks, not production-scale evidence.

The [larger benchmark](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md) includes a runnable experiment and
one current reference CSV. Re-run on October 18 against the standalone solver,
all six trials were feasible, but the best process gap was **17.82%**, versus
**0.00474%** for a simple greedy baseline. This demonstrates multicore execution,
not competitive solution quality or fixed-work speedup.

## Quality checks

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest src/rko_solver/tests -q
```

GitHub Actions runs the full suite on Linux and Windows. Tests include real
spawn execution, shared elites, independent RNG streams, algorithm execution,
callback exceptions, abrupt exits, blocked callbacks and interruption cleanup.
Property-based tests use deterministic generated cases for pool, key-vector,
validation, Q-learning and solver-result invariants.

Scoped mutation testing is on demand because it reruns tests for generated code
mutations. It covers selected high-risk components listed in `pyproject.toml`;
surviving mutants are review findings, not a CI score gate. The mutation run
excludes the runtime-Protocol conformance test because mutmut instrumentation
adds methods to Protocols; the test remains in the normal CI suite:

```bash
uv run --locked mutmut run
uv run --locked mutmut results
```

## Remaining release work

The library is **not yet release-ready**. See [contributor release gates](https://github.com/MarcosPTProenca/rko-solver/blob/main/CONTRIBUTING.md#before-a-release)
and the [C++/native evaluation criteria](performance.md). [Current limitations](status.md)
for the outstanding scheduling, parity, results, documentation and licensing work.
[C++ compatibility](cpp-compatibility.md) distinguishes verified reference
behavior from deliberate corrections and still-unimplemented functionality.
Installation and correctness checks are not evidence of competitive optimization.
Internal audits, agent plans and migration diaries are local working material,
not maintained public documentation.
