# Architecture

[Documentation index](index.md) · [Contributing](https://github.com/MarcosPTProenca/rko-solver/blob/main/CONTRIBUTING.md)

RKO has one public Python API and three implementations of the algorithms. You write the
problem in Python (decoder, objective, data reading, validation, reports); an **engine**
runs the search.

```text
your code
   |  RKO(problem, engine=..., execution=...)        src/rko_solver/solver.py
   v
public API: rko_solver.RKO, rko_solver.Callbacks | Numba | Blocks, rko_solver.problems.*, RKOResult, reports
   |
   v  engine selector (orchestration/engine.py, orchestration/auto.py for engine="auto")
   +--------------------+---------------------------+---------------------------+
   | python             | rust                      | cpp                       |
   | src/rko_solver            | engines/rust/rko-core     | engines/cpp (vendored     |
   | metaheuristics/    | engines/rust/rko-py       | C++ v2.0 headers + driver)|
   | components/        | src/rko_solver/engines/rust/     | same Python adapter       |
   | orchestration/     | (adapter)                 | as rust                   |
   +--------------------+---------------------------+---------------------------+
          parameters: src/rko_solver/parameters/*.toml, loaded and validated once in Python
```

## Where things live

| Path | What it is |
| --- | --- |
| `src/rko_solver/` | The public package and the **Python engine**. `import rko_solver` and `rko_solver.metaheuristics` live here, so the engine does not move. |
| `src/rko_solver/problem.py`, `problems.py` | Problem definitions accepted by `RKO`: `Callbacks`, `Numba`, `Blocks`, built-ins. |
| `src/rko_solver/orchestration/` | Validation of settings, process/thread/sequential execution, `engine="auto"`. |
| `src/rko_solver/engines/rust/` | Python side of the compiled engines: `runner.py` (calls the extension for both `rust` and `cpp`), `batch.py` (block objectives), `compiled.py` (Numba objectives). |
| `src/rko_solver/parameters/` | TOML defaults and adaptive grids: the single source of truth for every engine. |
| `engines/rust/rko-core/` | Pure Rust engine: 11 algorithms, Q-learning, shared elite pool, scheduler, built-in problems. No dependencies, no `unsafe`. |
| `engines/rust/rko-py/` | PyO3 bindings that build the `rko_solver._native` extension. |
| `engines/cpp/` | The `rko-cpp` crate: wrapper, driver and the vendored RKO C++ v2.0 headers (`vendor/`, unchanged, MIT). Compiled only with `RKO_BUILD_CPP=1` on Linux. |
| `engines/THIRD_PARTY_LICENSES.txt` | Inventory of the Rust crates linked into the extension, checked in CI. |
| `benchmarks/` | Reproducible measurements and the studies that produce them. |
| `tools/` | Release validation, license inventory and installed-artifact smoke checks. |

## How a solve flows

1. `RKO(...)` stores settings; nothing is read or started.
2. `solve(time_limit)` validates `engine`, `execution` and the problem, then loads the TOML
   parameters and applies overrides in Python.
3. `python` runs the algorithms in-process, threads or worker processes. `rust` and `cpp`
   receive plain numeric parameter snapshots through `NativeRunner` and run the portfolio inside
   the extension; Python callbacks are called back under the interpreter lock.
4. Every engine reports the same `RKOResult` and `SearchMetadata` records. `engine="auto"` runs
   short probes of the candidates that can run the problem and then the full budget on the winner.

## Changing the code

- A new **parameter default** or adaptive grid: edit the TOML; all engines read it.
- A new **algorithm**: write it in the Python engine first (reference), then port it to
  `engines/rust/rko-core`. The C++ engine only gets what the upstream reference publishes.
- A new **built-in problem**: implement the `Problem` trait in `rko-core` and expose it through
  `rko-py` and `rko_solver.problems`.
- Native changes need `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --release
  -- -D warnings` and `cargo test --release`; see [Rust and C++ engines](native.md).

## Deprecated import paths (removed in 0.4)

`rko_solver.batch`, `rko_solver.compiled`, `rko_solver.orchestration.native` and `rko_solver.native` re-export the modules that
moved to `rko_solver.engines.rust` and `rko_solver.problems` and emit a `DeprecationWarning` on import.
