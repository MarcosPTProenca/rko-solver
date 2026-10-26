# Contributing

Contribute to `MarcosPTProenca/rko-solver`, branch `main`. This is the standalone
library. Read [AGENTS](AGENTS.md)
for repository coding conventions and [current limitations](docs/status.md) before
interpreting a passing test suite as release or scientific validation.

## Development setup

The [architecture guide](docs/architecture.md) maps the public API, the three engines
(`src/rko_solver`, `engines/rust`, `engines/cpp`) and where to change what.

Use Python 3.14 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/MarcosPTProenca/rko-solver.git
cd rko-solver
uv sync --locked --group dev
uv run --locked pre-commit install
```

Run tools from the repository root. NumPy is the only runtime Python dependency;
the native engines in `engines/` needs a Rust toolchain at build time. `uv sync`
rebuilds the extension when Rust sources change. Keep callback/result contracts
and packaged TOML/fixture resources intact.

## Validate a change

Start with tests covering the touched behavior, then run the complete checks on
an otherwise idle machine or CI:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --release -- -D warnings
cargo test --release
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python tools/quality_gate.py
uv run --locked vulture
cargo deny check
uv run --locked pytest src/rko_solver/tests -q
uv run --locked pre-commit run --all-files
uv build
```

The quality gate keeps files under 600 lines and functions under complexity 10, 100 lines and
6 parameters (Python and Rust, tests excluded), forbids explanatory comments inside functions and
stops `mypy` errors from growing per file. Code that predates these limits is recorded in
`.quality/*.json`: it may stay but never get worse, and a new violation always fails. After
improving a recorded function, run `uv run python tools/quality_gate.py --tighten` and commit
the smaller baseline. Do not regenerate the baselines with `--write-baseline` to make a failure
pass; refactor instead. `--skip-types` runs the fast subset used by the pre-commit hook.

Every Python block in the README and `docs/*.md` is executed by
`src/rko_solver/tests/test_documentation_examples.py`: complete programs (with an
`if __name__ == "__main__":` guard) run as scripts, fragments run in order on their page with
`decode`, `objective`, `n_variables` and Knapsack data predefined, and `RKO.solve` is capped
at half a second. A block that cannot run in CI carries an invisible comment right before its
fence, `<!-- doc-test: skip <reason> -->` or `<!-- doc-test: requires cpp|numba|plotly -->`.
`test_golden_runs.py` freezes seeded results of every algorithm on the reproducible
configurations; if a behavior change is intended, regenerate with
`RKO_UPDATE_GOLDEN=1 uv run pytest src/rko_solver/tests/test_golden_runs.py` and review the
diff of `golden/runs.json`.

[CI](.github/workflows/ci.yml) checks Ubuntu/Windows, benchmark diagnostics, and
isolated wheel/sdist installations using `tools/smoke_installed.py`. A source-tree
import alone does not prove an installed package works. CI currently uses Python
3.14; additional interpreter/platform support needs explicit coverage.

Documentation changes should keep relative links and runnable examples correct.
Keep the [README](README.md) as the short entry point and detailed contracts in
[the documentation index](docs/index.md). Use [the extension guide](docs/extending.md)
when implementing an algorithm or replacing operators. Update the
[parameter reference](docs/parameters.md) alongside defaults or validation changes.
Link new user-facing pages from an existing guide.

## Documentation website

Preview the same maintained Markdown with the built-in MkDocs theme:

```bash
uv run --locked --only-group docs mkdocs serve
```

Use the reported `/RKO/` URL. Docs tools are a separate locked group, not
runtime dependencies. Before a docs change, run:

```bash
uv run --locked --group docs pytest src/rko_solver/tests/test_documentation_site.py -q
uv run --locked --only-group docs mkdocs build --strict
```

Navigation is the publication allowlist; do not add ignored/internal material.
See [site maintenance and owner setup](docs/documentation-site.md) for file
boundaries, main-only Pages deployment and recovery. Do not advertise a live URL
until deployment succeeds, or confuse development docs with a stable release.

## Bugs and pull requests

For a bug report, include a minimal reader/decoder/objective, failing input or a
small shareable substitute, Python/NumPy/package versions, OS, engine, execution, algorithms,
parameters, seed, worker limits and the complete traceback. Remove private data
and credentials. Explain expected versus observed behavior. For process issues,
show the guarded entry point and whether callbacks are importable.

A pull request should explain **why** the change is needed, the public contract
it affects, tests actually run, and any remaining limits. Add a focused regression
test for a behavior fix. Do not change algorithm semantics or defaults silently;
record intentional differences from C++ and update relevant examples/docs.

## Performance and scientific evidence

Follow [the C++ performance target and acceptance gates](docs/performance.md).
Do not run competing timing experiments alongside an active benchmark or modify
another run's source, cache, output or processes. Use a separate revision/output
for new experiments. Keep raw failures and provenance; predeclare workloads and
comparison rules rather than selecting only favorable cases.

Microbenchmarks, fixed-work traces, quality-at-time results and memory probes
measure different things. Test durations are not performance evidence; RSS sums
are not unique physical memory. Do not infer general C++ parity from one kernel,
one instance or synthetic objectives. Reproducible measurement code and identified
reference observations belong in `benchmarks/`; local audits, plans and generated
analysis stay ignored under the repository's existing conventions.

## Before a release

A version in source is **not evidence of publication or stable readiness**.
The initial controlled candidate is a platform-wheel prerelease; follow
[RELEASING](RELEASING.md).
Before tagging/publishing each version:

- Preserve the attribution in [NOTICE](NOTICE)
  and review new third-party additions. Do not replace licenses, relicense historical
  artifacts or claim that a compiler port settles provenance.
- Define supported Python/platform versions and the public API/deprecation policy.
- Review API/error documentation, stopping limits and current compatibility gaps.
- Run source, wheel/sdist and clean-install checks for every supported target.
- Record a versioned changelog note and reproduce any published claims.
- Establish the release owner and authenticated publishing process; no new PyPI
  release or native-wheel support is implied by these instructions.

Do not represent this checklist as completed or add speed/platform badges before
the corresponding evidence and distribution path exist.
