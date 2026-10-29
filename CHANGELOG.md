# Changelog

Candidate entries describe source changes, not proof of an index upload. Published
versions must be verified against their tag, manifest and package-index metadata.

## Unreleased

### Added

- [Installation and version guidance](docs/installation.md) separates current source from
  published prereleases, optional extras, regular artifacts and free-threading limits.
- Development-only Import Linter contracts, enforced in continuous integration and pre-commit,
  keep installed library code independent of repository tools, deployment and research separate
  from the runtime, and direct modeling imports out of numerical implementation packages.
  Regression tests check the policy against isolated source trees. Existing model conversion and
  the public API are unchanged; see [import boundaries](docs/architecture.md#import-boundaries).

- [Convergence guarantees](docs/guarantees.md): the protected-exploration property, its
  `(1 - mu)^(p + m*G)` miss bound for BRKGA, and what it does not say. Two tests back it: the best
  cost never loses an improvement on any engine, and BRKGA's miss rate on a five-bit target stays
  under the bound.
- `RKO(..., initial_keys=...)` evaluates copied, validated random-key vectors once before search
  and offers them to the shared pool across Python, Rust and C++; `result.initial_keys` reports
  seeded evaluations separately, while total evaluation counts include them.

### Removed

- **Deprecated engine and execution aliases are removed without another deprecation period.**
  `backend=` is rejected by `RKO`, single-algorithm solvers and `run`; solver/result `backend`
  properties, `RKO.native`, `RKO.compiled`, `RKO.batched` and the four deprecated import shims
  are gone. Use canonical problem objects with `engine` and `execution`; see
  [the replacements](docs/architecture.md#canonical-public-api). The checkout Knapsack example
  also rejects `--backend`. Default engine, execution and search behavior are unchanged.
- **The example programs and the `rko-solver` command are no longer part of the distribution.**
  `rko_solver.examples` (the scheduling and Knapsack examples, `decision_trace`, `decoder_diagram`,
  `analysis_report`, `model_conversion`), its reference instances and the `rko-solver` console
  script are gone from the wheel and sdist. They now live in `examples/` at the repository root and
  run from a checkout, for example `uv run python -m examples.run_kp_cpp_instances
  examples/instances/kp50.txt`. `rko-solver-service` is unchanged. The release checks now reject an
  artifact that contains `examples/`.
- The service scaffold's `adapter.py` carries its own scheduling problem instead of importing
  `rko_solver.examples`, so a scaffolded service depends only on the installed `rko-solver`.

### Changed

- Public documentation is consolidated around installation, first solve, usage and explicit
  limits. The shorter README retains its complete data-loading/validation program and both
  tested single-algorithm forms, while linking optional workflows instead of duplicating them.
  Outdated alias, command, compiler, publication and qualification descriptions are corrected;
  the shared site banner links the installation guide and includes macOS functional checks.
  Historical measurements and numerical behavior are unchanged.
- Free-threaded CPython 3.14 source-install verification now runs the complete installed-package
  smoke, including Python sequential/thread/process execution and Rust callback/block solves.
  The smoke rejects fallback to an enabled global interpreter lock. Published wheel targets
  remain the six regular `cp314-abi3` platforms; no free-threaded wheel support or performance
  gain is implied.
- Repository-only benchmark scripts are grouped under `runtime`, `algorithms`, `engines`,
  `modeling` and `problems`; use `python -m benchmarks.<group>.<study>` from the repository root.
  The existing `tuning` and `self_evolve` groups stay in the project. Identified measurements move
  to `benchmarks/references/` without changing their bytes or historical provenance. The Knapsack
  comparison writes its default new output under `results/`; explicit output paths still work.
  Old flat benchmark module paths are not retained as aliases. Installed library behavior is
  unchanged; see [the benchmark layout](benchmarks/README.md#layout).

- BRKGA rejects settings where `int(p * pm)` is zero, in every engine, and adaptive grids skip
  those states. Such a configuration used to run without any fresh uniform sample, which removes
  the exploration the algorithm relies on. The packaged defaults and grids are unaffected; an
  override such as `p=15, pm=0.05` now raises `ValueError`.

### Fixed

- With `execution="process"` (the default), a callback that overruns the five-second search
  grace no longer discards every solution found so far: `solve` stops the workers and returns
  the best solution with a `RuntimeWarning`, as the sequential and thread executions already
  returned theirs. The stopped searches report `stop_reason="time_limit"` and zero evaluations.
  `TimeoutError` is still raised when no solution exists yet or startup and initialization
  exceed their watchdogs.

## 0.3.0a4 — prerelease changes

Everything merged since 0.3.0a3: one change to the packaged fixed simulated-annealing
parameters, opt-in experimental additions to the model and self-evolve previews, and repository
only tuning evidence. The default engine, execution and adaptive mode are unchanged.

### Changed

- **Packaged fixed simulated-annealing parameters changed** (`[offline]` in `sa.toml`) from
  `sa_max=1000, t0=1000000.0, alpha=0.99, beta_min=0.01, beta_max=0.05` to
  `sa_max=1027, t0=340.3863933617792, alpha=0.8601360548373771,
  beta_min=0.0035204893377555175, beta_max=0.008242788398191462`. This affects runs with
  `adaptation="fixed"` or `control=0` on every engine, and results for a given seed change.
  The default adaptive mode does not read these values and is unchanged. To keep the previous
  behavior pass the old values through `hyperparameters={"SA": {...}}`. The new values were
  chosen on training data only and confirmed on held-out instances of three synthetic problem
  families; see `benchmarks/README.md` for the evidence and its limits.
- Experimental self-evolve model campaigns now use a one-second end-to-end model profile that
  reserves finalization time inside it. Records written with an older timing profile are
  rejected rather than silently migrated.

### Added

- `RKO.from_model(..., construction=callable, construction_deadline=...)` (development preview):
  an explicit, trusted construction callable `decode(keys, instance)` that returns complete
  numerical values, while the original model validation and scoring stay authoritative. Results
  of constructed models carry optional `statistics` (completed evaluation and infeasibility counts
  and capped distinct-output identities). Custom construction supports portable assignments but
  not native write-back, which raises `NotImplementedError`. Ordinary model and callback runs and
  the default result JSON are unchanged. See the modeling guide.
- `rko_solver.experimental.self_evolve` (opt-in, experimental): a review-gated Pi proposal
  transport that targets built-in OpenAI Codex OAuth on Linux/WSL, requires exact-source human
  approval before evaluation, and fails closed on unknown usage or cleanup. It has no API-key
  fallback and reports monetary cost as unknown; tests use a fake process, so live account
  access and provider terms are not qualified. An experimental bridge also lets isolated workers
  run generated assignment-construction programs against owned models. Its small fixtures check
  correctness only and are not evidence of optimizer value.

### Fixed

- A model job's end-to-end deadline could be renewed because the solver starts its search clock
  after setup. Construction-backed search and initialization sessions are now clamped to the
  absolute deadline while keeping the first-completion grace.

### Repository evidence (not in the wheel)

- `benchmarks/tuning` compares four parameter tuners with held-out scoring, and
  `benchmarks/README.md` reports a bake-off, a wider-range fresh-cohort study, and a
  cross-family fixed simulated-annealing preset selected on training data and confirmed on
  held-out instances on the Rust, Python and C++ engines. Every study states its limits; the
  evidence covers three synthetic families and says nothing about adaptive-mode quality. The
  optional `tuning` dependency group installs its extra libraries.
- The Rust-over-Python throughput gate measures simulated annealing with the previous parameters
  so its per-CPU reference stays valid.

## 0.3.0a3 — prerelease changes

Everything merged since 0.3.0a2: new opt-in features, a stricter test and quality regime, and
fixes found by it. The default engine and execution are unchanged.

### Added

- `max_evaluations` (RKO and every single-algorithm solver): stops each algorithm after exactly
  that many completed evaluations on the Python, Rust and C++ engines, with the new search stop
  reason `max_evaluations` and `RKOResult.max_evaluations`. Reproducible only where no
  concurrent pool exchange exists; see the usage guide.
- `RKO.from_model` (development preview): derives decoders and objectives from bounded linear
  models written with Pyomo, PuLP or OR-Tools MathOpt, installed through the `pyomo`, `pulp` and
  `mathopt` extras. See the modeling guide for what is supported and rejected.
- `rko_solver.experimental.self_evolve` (opt-in, experimental): bounded research that improves an
  existing seed decoder through an explicit file exchange. It adds no core dependency and is not
  an autonomous provider integration.

### Changed

- A non-finite objective now raises `ValueError("Objective callbacks must return finite costs.")`
  on the Python, Rust and C++ engines. The compiled engines used to raise a `RuntimeError` with a
  differently worded message; code that caught that `RuntimeError` must catch `ValueError`.
- The C++ engine honors `history_limit`: it used to accept the setting and return an empty
  history, and now records the same capped incumbent history as the Python and Rust engines.
- The NumPy lower bound is now `>=2.3.2`, the oldest release that ships a Python 3.14 wheel; the
  declared `>=2.0.0` could not be installed on the required Python and was never tested.
- `time_limit`, `auto_time` (one year), `max_evaluations`, `pool_size` and the population sizes
  have upper bounds, plus a bound on population times dimension, rejected with a `ValueError`
  before any engine starts.

### Fixed

- `time_limit=1e300` made the Rust engine panic, and a population of 10^12 tried to allocate
  terabytes; both are now rejected up front (see the bounds above).
- The Rust LNS and SA algorithms could run past their time limit when their temperature reached
  zero, because the budget was only checked in the inner loop; both now check it in the outer loop.

### Quality and test infrastructure

- Ratcheted structure, comment and type gates, a dead-code check, dependency audits and a
  Rust license and advisory check. Python and Rust coverage floors, a Rust-over-Python throughput
  ratio gate per CPU model, differential and golden-run suites across engines, a parity
  contract, property and fuzz tests, a weekly mutation-testing report, declared-minimum and
  macOS jobs, installed-wheel smoke tests of every optional extra on each platform wheel, and
  executed documentation examples. None of this ships in the wheel.

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
