# Rust and C++ engines

`engine="rust"` runs the eleven algorithms, Q-learning control and the shared elite
pool in a compiled Rust extension (`rko_solver._native`, sources in `engines/`). The Python API,
TOML parameter files, hyperparameter overrides and result metadata are unchanged: Python
still loads and validates every parameter and only passes numeric vectors to the engine.

Measured results against the pinned C++ reference and the Python engine, with their
limits, are in [benchmarks](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md#native-engine-versus-c-and-python).
They are not a general speed guarantee.

## Four ways to define a problem

Pick by how much of the objective you can compile or vectorize. Evaluations per second
below are 2-second SA/BRKGA/ILS runs on one 500-item Knapsack, same machine, same
algorithms ([study](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md#native-engine-versus-c-and-python) has the
controlled comparison):

| Style | You write | Interpreter lock | Evaluations (millions) |
| --- | --- | --- | ---: |
| Python callbacks | `decoder`, `cost` (any Python) | held per call | 0.04 |
| Block objective | one NumPy function over a `(rows, n)` array | held per block | 0.5 to 3.3 |
| Compiled objective | one Numba `@njit` function over one key vector | **released** | 0.8 to 8.7 |
| Native problem | implement `Problem` in `engines/rust/rko-core` | **released** | 0.8 to 6.6 |

**Python callbacks (compatible).** The decoder and objective you already have:

```python
result = RKO(
    rko_solver.Callbacks(decoder, cost, n_variables), engine="rust", random_state=42
).solve(10.0)
```

Each call receives a fresh NumPy array, exceptions raised in callbacks propagate
unchanged, and non-finite objectives fail the solve. Callbacks execute under the
interpreter lock, so worker threads take turns: the engine removes its own overhead
(startup, framework bookkeeping) but cannot add parallelism to Python code.

**Block objective (vectorize).** One function receives a fresh `(rows, n)` float array and
returns one finite cost per row (a NumPy array or a list):

```python
def batch_objective(keys):
    chosen = keys > 0.5
    return 100000.0 * np.maximum(0.0, chosen @ weights - capacity) - chosen @ profits

result = RKO(
    rko_solver.Blocks(batch_objective, n_variables, batch_size=64), engine="rust", random_state=42
).solve(10.0)
```

The engine fills blocks with candidates that are independent (population offspring,
IPR neighbors, LNS repairs, GRASP line searches, Nelder-Mead reflection/expansion/
contraction points) and, for sequential first-improvement local-search sweeps, scores
blocks of upcoming moves speculatively: when a move is accepted the rest of the block is
scored again against the new base, so the accepted trajectory is the sequential one and only
those stale scores are wasted. SA consumes blocks of neighbors up to the first acceptance.
`batch_size` (1 to 4096, default 64) bounds the work of one call, which cannot be
interrupted. The Python engine accepts the same function and calls it with one row.

**Compiled objective (Numba).** An `@njit` function of one key vector; the engine calls its
machine code from every worker thread without the interpreter lock:

```python
import numba

@numba.njit
def objective(keys):
    profit, weight = 0.0, 0.0
    for i in range(keys.shape[0]):
        if keys[i] > 0.5:
            profit += profits[i]
            weight += weights[i]
    return 100000.0 * max(0.0, weight - capacity) - profit

result = RKO(rko_solver.Numba(objective, n_variables), engine="rust", random_state=42).solve(10.0)
```

The function must be pure: reentrant, free of Python objects, never raising, and must not
write through or keep the key array (it is a view of engine memory during the call). A
`numba.cfunc` with signature `float64(CPointer(float64), intp)` is also accepted.
Compilation happens before the timed search. Install Numba with
`pip install "rko-solver[compiled]"`; it is optional.

**Native problem (fastest to start).** The decoder and objective run inside Rust. Python
exposes a built-in 0/1 Knapsack as an example (`RKO(rko_solver.problems.Knapsack(...), engine="rust")`); the
Rust core also contains TSP and Rastrigin problems used by the benchmarks. Built-in and
Numba problems run on the `rust` and `cpp` engines only; `engine="python"` rejects them. A new native
problem means implementing the `Problem` trait in `engines/rust/rko-core` (optionally
`evaluate_batch` and the `Incremental` trait); there is no Python-level API for defining one.

## Behavior to know

- `n_jobs` sets Rust worker threads. By default problems that run without the interpreter
  lock (native, Numba) use one thread per algorithm, as the C++ driver does. Python callbacks
  and NumPy block objectives hold the lock, so ten threads only fight for it: they default to
  **one** thread that runs the algorithms one after another, each with an equal share of the
  remaining time. On a ten-algorithm portfolio at 2 s, one thread beat ten for NumPy block
  objectives on TSP (36.3k against 51.1k tour length), Rastrigin (0.5 against 136) and
  Knapsack, and tied for scalar callbacks. An explicit `n_jobs` always wins, and with fewer
  threads than algorithms each worker splits its remaining time among its searches.
- Seeds make a single-worker run reproducible only up to timing, as in every engine.
  Random streams are the engine's own, so trajectories differ from the Python engine.
- `max_evaluations=N` stops each search after exactly `N` completed evaluations (block
  objectives score only the rows that fit before reporting the stop); the time limit still applies.
- `prefill_pool=True` fills the pool with `pool_size` random evaluated vectors first and
  reports it as `PoolInitializationMetadata`; the default stays off.
- Nelder-Mead takes its two pool vertices from the best pool entries for SA, GA, LNS, PSO
  and VNS (and may return one of them when better than the incumbent), and from two
  random entries for BRKGA, BRKGA-CS, GRASP and ILS. The choice came from a Knapsack plus
  TSP screen; the Python engine always uses the best entries and the C++ reference random
  entries.
- Local-search sweeps accept an improving move immediately and continue from it
  (strategy 1, the C++ benchmark setting); the Python engine's RVND uses fixed-base
  best improvement.
- The descent used by ILS, VNS, GRASP and BRKGA-CS tries the neighborhoods in the fixed
  order mirror, Farey, Nelder-Mead, swap and returns to the first after each improvement.
  The reference draws the next neighborhood at random; the fixed order, with the quadratic
  swap sweep last, reached better optima with fewer evaluations on Knapsack, TSP and
  Rastrigin calibration instances and removed a large TSP deficit
  ([study](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md#what-changed-after-the-first-held-out-round)).
- IPR seeds an empty pool up to its capacity with random evaluated vectors before relinking,
  so it works as a single algorithm; inside a portfolio peers fill the pool first.
- `verbose=True` prints lifecycle lines only; live pool progress is not available.
- Windows, Linux and macOS builds use the same sources; only Linux has measurements.

## Optional C++ reference engine

`engine="cpp"` runs the original RKO C++ v2.0 algorithms (vendored unchanged under
`engines/cpp/vendor`, MIT license, see [NOTICE](https://github.com/MarcosPTProenca/rko-solver/blob/main/NOTICE))
behind the same problem types as the Rust engine: built-in problems, Python callbacks,
`rko_solver.Numba` objectives and the full parameter, Q-learning and result interface. It exists
to compare engines on identical objectives and for users who want the published
algorithms exactly. It is not compiled by default, so a normal install needs no C++ compiler
or OpenMP:

```bash
RKO_BUILD_CPP=1 uv sync        # Linux, g++ with OpenMP (libgomp)
```

```python
from rko_solver import RKO

solver = RKO(problem, engine="cpp", random_state=7)
result = solver.solve(10)
```

Package managers cache built wheels without looking at `RKO_BUILD_CPP`; install with
`uv pip install --no-cache` (or `pip install --no-cache-dir`) when switching it. Check
`rko_solver._native.cpp_available()` to see whether a build includes it; without it,
`engine="cpp"` raises an error that names `RKO_BUILD_CPP`.

Differences from the Rust engine, all imposed by the reference code:

- Budgets are at least 0.05 seconds (the reference counts whole seconds, so a fractional budget
  is a shortened last second); there is no worker cap (`n_jobs` is rejected) because the
  reference runs one OpenMP thread per algorithm and, as in the original, stops all of them
  when the deadline passes.
- IPR needs at least 10 keys: the reference computes its block size as `dimension / 10` and
  crashes the process with a division by zero below that, so the wrapper rejects it first.
- Runs are serialized inside a process (the reference keeps its elite pool in a global).
- `max_evaluations` is enforced by the driver without touching the reference headers: the stop
  flag they poll also reports a per-thread limit, and the first objective call after the limit
  throws a private exception that ends that algorithm (the reference's IPR polls only its own
  timer). Counts are exact, and algorithms that reach their limit stop while the others go on.
- Scalar history and trajectory telemetry are not recorded, and every search reports the shared engine seed; the reference
  also draws part of its local-search neighborhood from an unseeded `rand()`, so equal
  `random_state` values do not give equal results.
- Python callbacks run on the reference's threads and need the interpreter lock, so a
  callback portfolio is serialized, as with the Rust engine.

## Automatic engine

`engine="auto"` chooses the engine for your specific problem by running short probes of every
candidate that can run it, then runs the full budget on the winner:

```python
from rko_solver import RKO

solver = RKO(
    rko_solver.Callbacks(decoder, cost, n_variables), engine="auto", auto_time=2.0, random_state=7
)
result = solver.solve(30)
print(result.engine, result.execution, result.auto.chosen, result.auto.reason, result.auto.scores)
```

- **Candidates.** Python callbacks: `rust`, `rust-threads` (all CPUs), `process` (the Python
  engine with process execution) and, when built, `cpp`. NumPy block objectives: `rust`,
  `rust-threads`, `cpp`. Built-in problems and Numba objectives: `rust`, `cpp`. Restrict them
  with `auto_candidates=["rust", "process"]`.
- **Probes.** Every candidate runs twice with the same two seeds; each probe gets
  `auto_time / (2 * candidates)` seconds (at least 0.05). `auto_time` is separate from the
  `time_limit` of the final run and defaults to 10 percent of it. The selection takes longer
  than `auto_time` when `process` is a candidate, because starting worker processes is not part
  of a probe's budget but is real time; `result.auto.selection_seconds` reports it.
- **Decision.** The lowest mean probe cost wins, but the preferred candidate (the order above)
  is kept unless the winner beats it by more than the probe noise (half the spread of its two
  probes) and by more than 5 percent of its cost. A short probe can favor a candidate that loses
  over the full budget, so leaving the preferred one needs a clear advantage.
- **Result.** The returned solution is the best of all probes and the final run; `result.engine` and
  `result.execution` name the executed engine and `result.auto` records every probe, the scores, the margin, the
  reason and the seconds spent. A candidate that fails (for example unpicklable callbacks for
  `process`) is excluded and its error is kept; if all fail, `RuntimeError` is raised.
- **Not reproducible bit for bit.** The choice depends on wall-clock timing, so equal
  `random_state` values can select different engines. `n_jobs` cannot be combined with auto.

**What the evidence says.** On 15 held-out instances of five problem types (two scalar-callback
problems, a heavy NumPy objective, a heavy pure-Python objective and a Numba objective, 10 s
searches) the choice was within one percentage point of the best fixed candidate in 39 of 45 runs
(median regret 0, mean 0.6 points), and it never selected `process`. But the selection time is
not free: at equal total compute time `auto` was worse than simply running `rust` on 12 of
15 instances (exact signed-rank p = 0.008), because the median 2.5 s of probing (26 percent of the
search budget) buys nothing when the preferred candidate wins anyway. Use `auto` when you do not
know which engine suits an objective and the budget is long compared with the selection; for
problems like those measured, `engine="rust"` is already the right choice. Details and losses:
[benchmarks](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md#automatic-engine).

## Limits

- A portfolio of **scalar Python callbacks** is worse on the Rust engine than on
  `execution="process"` (0 of 5 instances better in the study): the callbacks hold the
  interpreter lock, so the portfolio runs one search at a time while process execution uses
  one interpreter per worker. Single algorithms are better on the Rust engine, and a
  vectorized or Numba objective removes the gap.
- NumPy block objectives win for some algorithms (SA on Knapsack) but lose to the C++
  reference for the TSP and Rastrigin portfolios; for performance prefer a Numba objective or
  a native problem.
- Against the pinned C++ reference on held-out Knapsack, TSP and Rastrigin instances at 2 s,
  no algorithm was significantly worse and the default portfolio was better on Knapsack and
  TSP and tied on Rastrigin; IPR, MultiStart and some single algorithms are ties
  ([study](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md#native-engine-versus-c-and-python)).
- Only Knapsack, TSP and Rastrigin were measured, on one Linux (WSL2) machine.
- A compiled objective is an unsafe boundary: an impure, non-reentrant or raising function
  is undefined behavior, not a Python exception.
- PSO scores one particle per call even with a block objective, because each particle uses
  the global best updated by the previous one.

## Build and test

`uv sync` builds the extension (a Rust toolchain is required). After changing Rust code:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --release -- -D warnings
cargo test --release
uv run pytest src/rko_solver/tests/rko/test_rust_engine.py -q
```
