# Changelog

Candidate entries describe source changes, not proof of an index upload. Published
versions must be verified against their tag, manifest and package-index metadata.

## Unreleased

- The NumPy lower bound is now `>=2.3.2`, the oldest release that ships a Python 3.14 wheel; the
  declared `>=2.0.0` could not be installed on the required Python and was never tested.

- Upper bounds on `time_limit`, `auto_time` (one year), `max_evaluations`, `pool_size` and the
  population sizes, plus a bound on population times dimension, rejected with a `ValueError`
  before any engine starts. Previously `time_limit=1e300` made the Rust engine panic and a
  population of 10^12 tried to allocate terabytes.
- `max_evaluations` (RKO and every single-algorithm solver): stops each algorithm after exactly
  that many completed evaluations on the Python, Rust and C++ engines, with the new search stop
  reason `max_evaluations` and `RKOResult.max_evaluations`. Reproducible only where no
  concurrent pool exchange exists; see the usage guide.

## 0.3.0a2 — prerelease changes

First prerelease intended for PyPI, distributed as `rko-solver` and imported as `rko_solver`.
It contains everything listed below under 0.3.0a1, which was uploaded to TestPyPI only: its
after-upload check could not find the `rko-solver` command because the installed-index smoke
script searched the caller's `PATH` instead of the tested environment. The script now looks
next to the tested interpreter. See the [release procedure](RELEASING.md).

## 0.3.0a1 — TestPyPI only, superseded

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
