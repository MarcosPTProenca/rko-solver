# Installation and version boundaries

[Documentation index](index.md) · [First solve](getting-started.md) · [Current limitations](status.md)

These guides describe development `main`. A version string in `pyproject.toml`
can remain unchanged while development moves ahead; it does not identify the same
code as an already published archive. Record both the package version and source
revision when reproducing a development run.

## Choose your version

| Goal | Installation | Documentation to follow |
| --- | --- | --- |
| Use the code described by this development site | Clone the repository and use its lock file | This site and the same checkout's README |
| Use an immutable published prerelease | Pin an exact package version | That release's tagged README and changelog |

On 2026-10-29, the [official package metadata](https://pypi.org/pypi/rko-solver/json)
identifies **0.3.0a4** as the latest published prerelease, uploaded on 2026-10-28.
Its [release page](https://pypi.org/project/rko-solver/0.3.0a4/) lists six regular
`cp314-abi3` built distributions and one source distribution. It is an alpha
release, not a stable compatibility promise.

Development `main` has later changes: caller-provided `initial_keys`, repository-root
examples, removal of the installed demo command, and removal of deprecated solver
aliases. The published `0.3.0a4` does not accept `initial_keys`; it still accepts
`backend=` and contains `rko_solver.examples`. Do not copy that older layout into
new development examples.
See the [development changelog](https://github.com/MarcosPTProenca/rko-solver/blob/main/CHANGELOG.md)
and [release-tagged README](https://github.com/MarcosPTProenca/rko-solver/blob/v0.3.0a4/README.md).

## Install the development checkout

Use Python **3.14 or newer**, [uv](https://docs.astral.sh/uv/) and a
[Rust toolchain](https://rustup.rs/). The declared minimum is Python 3.14;
functional checks currently cover CPython 3.14, not every future interpreter
accepted by the metadata.

From a terminal:

```bash
git clone https://github.com/MarcosPTProenca/rko-solver.git
cd rko-solver
uv sync --locked --no-dev
uv run --locked --no-dev python -m examples.instance_pipeline
```

The last command loads the supplied scheduling instance, solves it and checks the
returned solution. It needs no private input file or optional dependency. For your
own problem, follow [the first solve](getting-started.md) or the
[file-to-solution tutorial](problem-adapters.md).

Run checkout commands from the repository root. `--no-dev` avoids development test
tools. Contributors instead install the development groups described in
[CONTRIBUTING](https://github.com/MarcosPTProenca/rko-solver/blob/main/CONTRIBUTING.md).
The native extension is built during source installation even when you select the
Python search engine, so this source workflow requires Rust.

## Use a published package

The distribution name is `rko-solver`; the Python import is `rko_solver`.
The PyPI project named `rko` is a different package.

For an isolated check of the published prerelease, without installing the checkout:

```bash
uv run --isolated --no-project --python 3.14 --with "rko-solver==0.3.0a4" python -c "from importlib.metadata import version; print(version('rko-solver'))"
```

For your own script, replace the final `-c` expression with the script's path.
A matching prebuilt distribution needs no compiler. If no artifact matches your
interpreter, architecture or system library, installation builds the source
archive and needs Rust instead. Keep published-package runs separate from an
editable development environment when comparing behavior.

## Optional features

NumPy is the only required Python runtime dependency. Install only the extras you
use; no reporting, model frontend, service framework or agent account is required
for ordinary optimization.

| Extra | Purpose | Guide |
| --- | --- | --- |
| `compiled` | Numba-compiled objectives | [Native problem styles](native.md) |
| `reports` | Offline interactive reports | [Reports](reporting.md) |
| `pyomo`, `pulp`, `mathopt` | A selected bounded-model frontend | [Model conversion](modeling.md) |
| `service` | Private supervised HTTP service and scaffold | [Service](service.md) |

From the development checkout, select extras on the command that uses them:

```bash
uv run --locked --no-dev --extra reports python -m examples.analysis_report
uv run --locked --no-dev --extra pyomo python -m examples.model_conversion pyomo
```

Add repeated `--extra` options to combine features. `uv run` synchronizes its
requested environment: an earlier `uv sync --extra reports` is not a substitute
for selecting `--extra reports` on a later run. For a published package, pin the
version and select extras in its requirement, such as
`rko-solver[reports]==0.3.0a4`; do not assume the development examples ship with it.

Documentation tools are separate from runtime extras. Preview with
`uv run --locked --only-group docs mkdocs serve`; see
[site maintenance](documentation-site.md) for validation and publication rules.

## Platforms, C++ and free threading

The regular release policy covers Linux x86_64/aarch64, Windows x86_64/ARM64 and
macOS x86_64/arm64. Its `cp314-abi3` tag uses the stable application binary
interface; it is **not** a promise that the same artifact works on free-threaded
CPython. Other systems may need a source build and are not certified by that
platform matrix.

Linux release artifacts include the optional C++ reference engine. Source builds
require explicit `RKO_BUILD_CPP=1` to include it; non-Linux release artifacts do
not include it. See [build requirements and availability checks](native.md#optional-c-reference-engine).
Selecting the Python engine does not require a separate C++ installation.

Free-threaded CPython 3.14 is checked on Linux through a **source installation**
and the full installed-package smoke. That check rejects an enabled global
interpreter lock before and after execution. It does not certify all optional
extras, free-threaded artifacts on every platform, arbitrary callback thread
safety or a throughput improvement. Manage your callbacks' mutable state yourself.

Optional native dependencies can enable the lock even on a free-threaded interpreter.
The pinned MathOpt extension in OR-Tools 9.15.6755 does not declare free-threading
safety and enables the lock by default when imported. Do not force the lock off to
silence that warning or make a check pass; use a qualified interpreter/dependency
combination instead.

If regular and free-threaded interpreters coexist, use an explicit interpreter
path when the distinction matters. Check the actual interpreter and resulting
artifact tags rather than inferring compatibility from a version-only selector.
See [the release policy](https://github.com/MarcosPTProenca/rko-solver/blob/main/RELEASING.md)
for approved targets and artifact verification.
