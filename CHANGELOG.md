# Changelog

Candidate entries describe source changes, not proof of an index upload. Published
versions must be verified against their tag, manifest and package-index metadata.

## 0.3.0a1 — prerelease changes

First intended prerelease, distributed as `rko-solver` and imported as `rko_solver`.
See the [release procedure](RELEASING.md).

- Distribution `rko-solver`, import package `rko_solver`, commands `rko-solver` and
  `rko-solver-service`.
- Explicit application-owned reader → bound decoder/objective → configured solver
  → `solve(time_limit=...)`, with immutable `RKOResult` and optional post-run diagnostics.
- RKO portfolios and eleven individual managed solver objects share the same
  constructor settings, execution implementation and nested hyperparameter mapping.
- Sequential/thread/process execution, owned random streams, parameter validation,
  bounded optional histories and documented cancellation/resource limitations.
- NumPy-only core; optional Plotly reports, post-run validation and static decoder
  diagrams with explicit provenance/privacy boundaries.
- Maintained development website and controlled platform-wheel distribution checks.
- Optional compiled Rust engine (`engine="rust"`) and optional C++ reference engine
  (`engine="cpp"`, bundled in the Linux wheels), with a generated statistical benchmark
  report; the default is unchanged (`engine="python"`, `execution="process"`).
- `engine="auto"` chooses among them by short probes on the user's own problem and records
  its decision in `result.auto`; it costs selection time and is not reproducible bit for bit.
- One constructor, `RKO(problem, engine=..., execution=...)`, where the problem is an
  `rko_solver.Callbacks`, `rko_solver.Numba`, `rko_solver.Blocks` or built-in `rko_solver.problems` object; the short form
  `RKO(decoder, cost, n_variables)` is unchanged. `execution` (`process`, `thread`,
  `sequential`) applies to the Python engine only. Results report `engine` and `execution`.
- Deprecated until 0.4, with a `DeprecationWarning`: `backend=` (and `RKOResult.backend`,
  `RKO.backend`), `RKO.native`, `RKO.compiled`, `RKO.batched` and the `rko_solver.native` module.
- Python >=3.14; release artifacts are CPython 3.14 abi3 wheels for Linux x86_64/aarch64, Windows
  x86_64/ARM64 and macOS arm64/x86_64, each installed and tested on its own platform, plus
  an sdist.
- Owner-confirmed current distribution rights/attribution, retained Apache-2.0 code
  license and MIT fixture license, with both texts included in artifacts.

No stable-release, general C++ performance-equivalence, universal interpretability or
complete feasibility certification is implied. The initial pipeline publishes only the
tested platform wheels and sdist; other platforms (musl, 32-bit) build from the sdist with a
Rust toolchain.
Historical release artifacts retain their original metadata/licenses.
