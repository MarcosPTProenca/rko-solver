# Extend the library

[Documentation index](index.md) · [Contributing](https://github.com/MarcosPTProenca/rko-solver/blob/main/CONTRIBUTING.md)

## First choose the smallest extension

- New optimization problem: implement a [decoder and objective](problem-adapters.md).
- Different settings: use [fixed overrides or adaptive TOMLs](parameters.md).
- Different refinement/perturbation: use the existing operator protocols.
- New search strategy: subclass `MetaheuristicBase` and implement the lifecycle.

Do not copy an entire algorithm to change a parameter or a local-search operator.

## Replace an operator

Advanced kernel constructors, explicitly imported from implementation modules,
accept `LocalSearchOperator` and, for SA/ILS/VNS,
`ShakingOperator`. A local-search callable accepts
`(keys, decoder, cost, *, pool)` and returns `(keys, objective)`. The `pool=` keyword
is part of the contract. It may be None. A shaking callable accepts
`(keys, beta_min, beta_max)` and returns the perturbed NumPy vector.

Use the same callback contract as the owning algorithm and return keys consistent
with the objective value. Preserve bounds and dimension; do not assume that
mutating an argument is harmless. Use owned RNGs if the operator is stochastic.
Do not silently suppress callback exceptions or report uncomputed objectives.

Public algorithm solver objects share RKO's constructor and `.solve(...)` contract,
not every low-level kernel argument. Operator overrides remain an advanced kernel
feature, with the lifecycle limitations documented in
[advanced kernel integration](standalone.md#advanced-kernel-integration).

## Implement a new metaheuristic

Read `src/rko_solver/metaheuristics/base.py` and a comparable implementation first.
A concrete class must:

1. Inherit `MetaheuristicBase`, retain `decoder`, `cost`, `tam_solution` and its pool,
   and validate fixed and adaptive settings before searching.
2. Implement `solve(limit_time)` with a timer, cooperative stop checks and consistent
   minimization comparisons. Respect cancellation and callback exceptions.
3. Return the low-level `(extra, best_keys, best_cost)` tuple, retaining a valid
   incumbent when interrupted. Publish detached improvements to the pool with
   `pool.insert((cost, keys), self.name)`.
4. Use `RandomStreams` for library randomness rather than reseeding global generators.
5. Describe strategy, pseudocode, acceptance/stopping rules, inputs, outputs and
   practical complexity in the module header; document every class/function with
   types, parameters, returns and applicable exceptions.

Use an existing algorithm module as a maintained implementation example rather
than a minimal skeleton that omits validation, RNG or stopping behavior.

To integrate with the configured high-level API, update the single `MH_SPECS`
registry, canonical aliases if needed, parameter-file mapping/axis schema,
packaged TOML defaults and public exports. To offer a dedicated managed public
solver, add a single-algorithm RKO subclass and its lazy package export; reuse
RKO.solve rather than copying execution logic. Registry membership alone does not
create that public solver class, and arbitrary subclasses are not auto-registered.
There is no third-party plugin-registration API in this change.

## Keep the three engines aligned

The Python, Rust and C++ engines share one public contract: the same `RKO(...)` settings, the
same errors and the same result metadata. A setting or an algorithm that only one engine
understands is a defect, not a feature, so extend all three or reject it explicitly in the
others with a `ValueError` that says so.

Where each piece lives:

| Piece | Python | Rust | C++ |
| --- | --- | --- | --- |
| Algorithm | `metaheuristics/` | `engines/rust/rko-core/src/mh/` | the vendored reference headers, driven by `engines/cpp/cpp/rko_cpp.cpp` |
| Search budget, history, cost checks | `orchestration/search.py` | `rko-core/src/search.rs` | `Decoder` in `rko_cpp.cpp` |
| Settings and errors to Python | `solver.py`, `orchestration/preparation.py` | `engines/rust/rko-py/src/lib.rs` | `engines/cpp/src/ffi.rs` |

The C++ headers are vendored unchanged and carry the MIT license; behavior is added around them
in the driver, never by editing them.

When you add a setting or an algorithm:

1. Read the setting once in `orchestration/preparation.py` so every engine validates it the
   same way, and pass it down to the Rust and C++ calls.
2. Add the algorithm to `Algorithm::all()` in `rko-core/src/mh/mod.rs`; the C++ driver
   identifies algorithms by their index in that list, so keep the order.
3. Report it in the per-search metadata with the same field names and stop reasons
   (`time_limit`, `max_evaluations`, `completed`, `cancelled`, `not_started`).
4. Pin it in the parity contract (`tests/test_engine_parity_contract.py`):
   run it on every available engine and compare reports and error messages exactly.
5. If it changes what an algorithm computes, refresh the golden runs
   (`test_golden_runs.py`) and say why in the pull request; they are the only frozen results.

What the contract pins today: parameters and overrides (including a custom TOML directory),
adaptive grids and controls, the report fields and stop vocabulary, bounded incumbent history,
and the error for a non-finite cost (`ValueError`, one message). The differential tests
(`test_engine_differential.py`) additionally check that the reported cost is recomputed from the
reported keys with independent code.

Differences that are intended, and documented, not bugs:

- **`verbose`:** the Python engine prints live progress; Rust and C++ print only a start and a
  done line, because progress would mean calling back into Python from the search loop.
- **C++ startup:** the reference always fills its pool first, so `initialization` is reported
  even without `prefill_pool`.
- **Workers and seeds:** `n_jobs` does not limit the C++ engine (one thread per algorithm), and
  its seeds are not reproducible (see [native engines](native.md)).

Before claiming parity for anything new, run the same solve on each engine and compare what
comes back; a probe like that is how the history and exception differences were found.

## Required tests and documentation

- Constructor and factory invalid scalar/joint-domain cases.
- Finite keys/cost consistency and short-budget incumbent preservation.
- Fixed and supported adaptive paths, RNG isolation, callback failures.
- Pool interaction and operator substitution where applicable.
- The common constructor/solve contract, signatures/defaults and result metadata.
- The same behavior on every engine, through the parity contract above.
- Public examples and parameter reference updated with the implementation.

Run from this repository root:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest tests -q
```

For public behavior changes, explain compatibility and any seeded-trajectory or
callback-count changes. Passing unit tests does not establish scientific parity,
optimization quality, or production release readiness.
