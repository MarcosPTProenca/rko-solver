# RKO

[![CI](https://github.com/MarcosPTProenca/rko-solver/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MarcosPTProenca/rko-solver/actions/workflows/ci.yml)

[Development documentation](https://marcosptproenca.github.io/rko-solver/)

A domain-neutral **random-key optimization** library with cooperating
metaheuristics, inspired by [RKO C++ v2](https://github.com/RKO-solver/RKO_Cpp_v2.0).
You provide a decoder and an objective to minimize; RKO searches for good key
vectors without assuming a problem format or solution representation.

## What you implement

**For optimization, you provide the data reader, decoder and objective function.
You may also provide `validate` for optional post-solve checks. RKO provides the
optimization algorithms; all of these are ordinary Python functions.**

| Your code | Responsibility |
| --- | --- |
| `read_data(source)` | Read your instance and return the data needed by your problem. Skip this function if the data is already in memory. |
| `decoder(keys)` | Use the loaded data to convert a key vector into your domain solution. |
| `objective(solution)` | Use the loaded data to return a finite scalar to **minimize**. This callable is passed as `cost` in the library API. |
| `validate(solution)` **(optional)** | Check your domain constraints after solving and return a `ValidationReport`. Omit it if you do not need final-solution checking. |

Load the data once in your application, then bind it to the decoder and objective.
RKO calls those two callbacks during search; it does **not** invoke `read_data`.
Function names are yours to choose. Module-level functions with `functools.partial`
or methods holding the data both work; **an adapter class is optional**, not a
required interface. Optional validation uses the Python API `assess_result(...)`,
not a CLI-only feature; it runs outside the search budget and never implicitly
reevaluates the objective.

You do **not** need to implement a metaheuristic, population, elite pool, local
search, shaking or Q-learning controller. Select an algorithm and configure values
such as `n_variables`, `time_limit` and `random_state`; the library supplies the
search machinery and default operators. Custom operators are advanced overrides,
not prerequisites. See the [complete data-loading example](https://marcosptproenca.github.io/rko-solver/problem-adapters/).

## What it provides

- Configurable portfolios: BRKGA, BRKGA-CS, GA, GRASP, ILS, IPR, LNS, MultiStart,
  PSO, SA and VNS; the packaged default selects ten algorithms.
- Sequential, thread and process execution on the Python engine with shared elites, owned
  library RNG streams, validated parameters and immutable result metadata.
- An optional compiled Rust engine (`engine="rust"`) that runs the same eleven
  algorithms and TOML configuration. Define the problem as Python callbacks
  (`rko_solver.Callbacks`), a NumPy block objective (`rko_solver.Blocks`), a Numba-compiled objective
  (`rko_solver.Numba`, evaluated without the interpreter lock) or a built-in problem
  (`rko_solver.problems`); `engine="auto"` picks the engine for your problem by short probes.
  See [Rust and C++ engines](https://marcosptproenca.github.io/rko-solver/native/).
- Python callbacks for arbitrary problems; **NumPy is the only required runtime dependency**
  (Numba is optional: `pip install "rko-solver[compiled]"`).
  Installing a prebuilt wheel needs no compiler; building from source needs a Rust
  toolchain. No C++ installation or agent framework is required.

## Install from source

Requires **Python 3.14+**, [uv](https://docs.astral.sh/uv/) and a
[Rust toolchain](https://rustup.rs/) (the native extension is built from source).
CI currently tests Python 3.14 on Ubuntu, Windows and macOS (x86_64 and ARM); other combinations are not
certified by that matrix.

```bash
git clone https://github.com/MarcosPTProenca/rko-solver.git
cd rko-solver
uv sync --locked --no-dev
```

Run Python scripts with `uv run --locked --no-dev python your_script.py` from
this checkout. For reproducible use, record the Git revision as well as the lock
file.

## First solve

This complete Python example loads a small instance, binds its data to your
functions, solves, and optionally checks the returned solution. No CLI or adapter
class is required. Replace the example's file format and domain logic with yours.

Save this as `instance.json` in the repository root:

```json
{
  "lower": [-5.0, -5.0, -5.0],
  "upper": [5.0, 5.0, 5.0],
  "target": [1.0, -2.0, 0.5]
}
```

Save this as `quickstart.py` beside it:

```python
import json
import numpy as np
from functools import partial
from pathlib import Path
from rko_solver import RKO
from rko_solver import ValidationCheck
from rko_solver import ValidationReport
from rko_solver import assess_result


def read_data(path: Path) -> dict[str, np.ndarray]:
    """Load the example's bounds and target once before searching.

    Args:
        path (Path): UTF-8 JSON instance file.

    Returns:
        dict[str, np.ndarray]: Equal-length lower, upper and target vectors.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If JSON, fields, dimensions or bounds are invalid.
        TypeError: If values cannot be converted into numerical arrays.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"lower", "upper", "target"}:
        raise ValueError("Expected lower, upper and target fields.")
    data = {name: np.asarray(value, dtype=float) for name, value in raw.items()}
    shape = data["lower"].shape
    if data["lower"].ndim != 1 or data["lower"].size == 0 or any(
        value.shape != shape or not np.isfinite(value).all() for value in data.values()
    ):
        raise ValueError("Expected nonempty, equal-length finite vectors.")
    if np.any(data["lower"] >= data["upper"]):
        raise ValueError("Each lower bound must be smaller than its upper bound.")
    return data


def decoder(keys: np.ndarray, *, data: dict[str, np.ndarray]) -> np.ndarray:
    """Map random keys into the loaded variable bounds.

    Args:
        keys (np.ndarray): Random-key chromosome.
        data (dict[str, np.ndarray]): Loaded instance data, bound before search.

    Returns:
        np.ndarray: Decoded numerical solution.
    """
    return data["lower"] + (data["upper"] - data["lower"]) * keys


def objective(solution: np.ndarray, *, data: dict[str, np.ndarray]) -> float:
    """Minimize squared distance to the loaded target.

    Args:
        solution (np.ndarray): Decoded numerical solution.
        data (dict[str, np.ndarray]): Loaded instance data, bound before search.

    Returns:
        float: Finite objective for this example's input.
    """
    difference = solution - data["target"]
    return float(np.dot(difference, difference))


def validate(solution: np.ndarray, *, data: dict[str, np.ndarray]) -> ValidationReport:
    """Optionally check final dimensions, finiteness and domain bounds.

    Args:
        solution (np.ndarray): Post-run reconstructed solution.
        data (dict[str, np.ndarray]): Same loaded instance used during search.

    Returns:
        ValidationReport: Declared feasibility for the implemented bound checks.
    """
    valid = solution.shape == data["lower"].shape and bool(
        np.isfinite(solution).all()
        and np.all(solution >= data["lower"])
        and np.all(solution <= data["upper"])
    )
    return ValidationReport(
        checks=(ValidationCheck(
            "variable_bounds", "passed" if valid else "failed",
            message="Check the declared shape, finite values and variable bounds.",
        ),),
        feasible=valid,
    )


def main() -> None:
    """Load once, optimize with data-bound callbacks, then optionally validate.

    Raises:
        OSError: If the instance cannot be read.
        ValueError: If the input or solver settings are invalid.
        TypeError: If instance values are not numerical.
    """
    data = read_data(Path("instance.json"))
    decode = partial(decoder, data=data)
    cost = partial(objective, data=data)
    solver = RKO(
        decoder=decode, cost=cost, n_variables=data["lower"].size,
        algorithms=["SA", "PSO"], adaptation="fixed",
        hyperparameters={
            "SA": {"sa_max": 100, "t0": 1000.0, "alpha": 0.98},
            "PSO": {"p_size": 20, "c1": 1.5, "c2": 1.5, "w": 0.6},
        },
        execution="sequential", random_state=42,
    )
    result = solver.solve(time_limit=1.0)
    assessment = assess_result(
        result, decoder=decode, validator=partial(validate, data=data),
    )
    print("Solution:", assessment.solution)
    print("Objective:", result.cost)
    print("Validation:", assessment.validation.status)
    print("Evaluations:", result.n_evaluations)


if __name__ == "__main__":
    main()
```

Run `uv run --locked --no-dev python quickstart.py` from that directory.
The wiring is explicit:

1. **`data = read_data(...)`** runs in your application before constructing RKO.
2. **`partial(decoder, data=data)`** and **`partial(objective, data=data)`** bind
   loaded data to the callbacks passed to `RKO(decoder=..., cost=...)`.
3. **`solver.solve(...)`** runs the selected metaheuristics using those callbacks.
4. **`assess_result(..., validator=partial(validate, data=data))`** optionally checks
   the post-run solution. Neither `read_data` nor `validate` runs inside search.

**To skip validation**, omit `validate` and all `assessment`-related lines. Use
`solution = decode(np.asarray(result.keys))` and print that solution instead.
Assessment validates a post-run reconstruction, not optimality or an original
execution witness. See [validation details](https://marcosptproenca.github.io/rko-solver/solution-validation/).

The decoder receives a one-dimensional NumPy key vector; the objective must
return a **finite scalar to minimize**. Readers and feasibility logic belong in
your problem adapter. `random_state` owns library draws, not randomness inside
user callbacks; timed runs can still differ because of scheduling.

The main API is `RKO(...)`: the example runs a cooperating SA/PSO portfolio with
shared elites. It explicitly selects sequential execution for a simple first run.
RKO's default is a **process portfolio**: use importable callbacks and a guarded entry point
as shown in the [process-safe example](https://marcosptproenca.github.io/rko-solver/usage/#spawn-compatible-callbacks).
Sequential execution gives **each selected algorithm** its own search allowance; process
startup is excluded, and opted-in pool prefill has a separate allowance. A callback
can overrun its deadline. `time_limit` is not a hard end-to-end timeout.

## Run one metaheuristic instead

Keep `read_data`, the data-bound `decode`/`cost` functions and optional assessment
unchanged. Replace only the solver block inside `main`.

**Through the main RKO API**, select one algorithm:

<!-- doc-test: skip fragment that replaces the solver block inside main() of the first program, whose names are local to main -->
```python
solver = RKO(
    decoder=decode, cost=cost, n_variables=data["lower"].size,
    algorithms=["SA"], adaptation="fixed", execution="sequential", random_state=42,
    hyperparameters={"SA": {"sa_max": 100, "t0": 1000.0, "alpha": 0.98}},
)
result = solver.solve(time_limit=1.0)
```

**Through the direct algorithm API**, import `SA` from `rko_solver.metaheuristics`
alongside the other imports, then use:

<!-- doc-test: skip fragment that replaces the solver block inside main() of the first program, whose names are local to main -->
```python
solver = SA(
    decoder=decode, cost=cost, n_variables=data["lower"].size,
    adaptation="fixed", execution="sequential", random_state=42,
    hyperparameters={"SA": {"sa_max": 100, "t0": 1000.0, "alpha": 0.98}},
)
result = solver.solve(time_limit=1.0)
```

Both forms use the **same configure-then-solve lifecycle** and return `RKOResult`.
`SA` selects SA automatically; `PSO`, `GRASP` and the other
[built-in classes](https://marcosptproenca.github.io/rko-solver/standalone/#algorithm-catalog) select their own algorithm.
They inherit RKO's execution, stopping, RNG ownership, history and `result_` behavior.
The examples explicitly select sequential execution; all public solvers otherwise share
RKO's default process execution and configured adaptation mode.

### Algorithm hyperparameters

`hyperparameters` is an explicit constructor parameter for **both** RKO and the
individual solver classes. Use the same nested dictionary: algorithm name -> scalar
overrides. It controls the algorithm itself, not worker or engine settings. Only keys
you provide replace defaults; manual scalar overrides require `adaptation="fixed"`.

| Algorithm | Configurable scalar keys |
| --- | --- |
| SA | `sa_max`, `t0`, `alpha`, `beta_min`, `beta_max` |
| ILS | `beta_min`, `beta_max` |
| VNS | `k_max`, `beta_min` |
| LNS | `beta_min`, `beta_max`, `t0`, `alpha_lns` |
| BRKGA / BRKGA-CS | `p`, `pe`, `pm`, `rhoe` |
| GA (`GeneticAlgorithm`) | `size_pop`, `prob_cros`, `prob_mut` |
| PSO | `p_size`, `c1`, `c2`, `w` |
| GRASP | `hs`, `he` |
| IPR / MultiStart | No configurable search scalars |

See the [parameter reference](https://marcosptproenca.github.io/rko-solver/parameters/) for meanings, defaults, joint
constraints and adaptive grids. Some identically named keys have different meanings
between algorithms (notably BRKGA versus BRKGA-CS `pm`); do not copy interpretations
blindly. Each dictionary entry must name an algorithm actually selected by the solver.

## Use your own instance data

The application loads data **once before searching** and binds it to the decoder
and objective. Metaheuristics do not need their own file readers:

```text
your read_data(path) -> loaded data
                    -> your decoder + objective (data bound once)
                    -> chosen algorithm or RKO (provided by the library)
                    -> best keys -> explicit final decoding
```

A complete [file-to-solution tutorial](https://marcosptproenca.github.io/rko-solver/problem-adapters/) shows a JSON
scheduling instance where the decoder uses loaded processing times and the objective
uses loaded due dates. The tutorial groups these two callbacks in an importable
class for convenience; you can use functions instead. The same callbacks work
unchanged with `SA(...).solve(...)`, other built-in classes and RKO portfolios. There is
no required base class.

Run the packaged example without creating an input file:

```bash
uv run --locked --no-dev python -m rko_solver.examples.instance_pipeline
```

For your own file, use the tutorial's schema or replace its reader/adapter with your
application's format. Loading once does not mean shared RAM across process workers;
keep callbacks pickleable and account for copied data when choosing `n_jobs`.

## Interactive analysis reports

Install the optional reporting dependency with `uv sync --locked --extra reports`.
To plot convergence, add `history_limit=1000` to the solve call above. Import
`OptimizationReport` from `rko_solver.reporting` alongside your other imports, then add
this after assessment inside `main`:

<!-- doc-test: skip uses result and assessment, which are local to main() of the first program -->
```python
report = OptimizationReport(result, assessment=assessment)
report.write_html("run.html")
report.write_json("run.json")
```

If you skipped optional validation, use `OptimizationReport(result)` instead.

The HTML embeds Plotly once and works offline with hover, zoom and trace selection.
Reports are generated after search, never by reevaluating callbacks. History is off
by default; compacted curves are labelled approximate. See [interpretation and
privacy boundaries](https://marcosptproenca.github.io/rko-solver/reporting/); automatic PDF export is not included.
Core optimization still requires only NumPy.

Reports can optionally include an [author-declared or source-derived decoder
diagram](https://marcosptproenca.github.io/rko-solver/decoder-diagrams/), explicitly attached through `decoder_diagram`.
Source inspection never runs automatically and static structure is not an execution
trace, a complete explanation or a feasibility/optimality certificate.

## Bounded mathematical models (development port)

The optional Pyomo, PuLP and MathOpt model converter is being ported to this
repository. It generates owned sparse callbacks with bounded continuous
feasibility restoration and independent feasible-witness evidence.

```python
import importlib.util
from rko_solver import RKO

if importlib.util.find_spec("pyomo") is not None:
    import pyomo.environ as p

    model = p.ConcreteModel()
    model.x = p.Var(bounds=(0, 1))
    model.y = p.Var(bounds=(0, 1))
    model.balance = p.Constraint(expr=model.x + model.y == 1)
    model.objective = p.Objective(expr=model.x + 2 * model.y)
    solver = RKO.from_model(model, engine="python", execution="sequential", repair="continuous")
    result = solver.solve(0.1)
    print(result.model.status)
else:
    print("Install rko-solver[pyomo] to run this optional example.")
```

Install the matching extra (`rko-solver[pyomo]`, `[pulp]` or `[mathopt]`). Integer
choices remain fixed during repair; this is not a complete MILP feasibility solver.
`repair="none"` retains box-only mapping. Rust/C++/automatic engine selection is
not supported for this factory because those paths lack its model-witness contract.

[Model contracts](https://marcosptproenca.github.io/rko-solver/modeling/) and
[historical qualification](https://marcosptproenca.github.io/rko-solver/model-validation/)
distinguish prior measurements from this renamed port. Local feature tests and mandatory
quality ratchets pass; fresh hosted-platform, artifact and performance qualification
remain pending.

## Optional observed decoder decisions

To retain decisions from the **original evaluated decoder invocation**, explicitly
emit `record_decision(action, reason)` inside your decoder and opt in with
`RKO(..., trace=TraceConfig())`, importing both helpers from `rko_solver.tracing`.
Existing callback signatures and uninstrumented usage remain unchanged.

Try the instrumented scheduling adapter and its optional offline report:

```bash
uv run --locked python -m rko_solver.examples.decision_trace
uv run --locked --extra reports python -m rko_solver.examples.decision_trace --html results/observed-decisions.html
```

`result.trace` distinguishes recorded, no-events, truncated and unavailable evidence.
Recording is bounded and off by default; it can be expensive for cheap callbacks and
uses the existing search budget. The engine never replays callbacks or borrows a
matching keys/cost trace when evaluation identity is lost. Observations are partial
author-emitted decisions, **not static flowcharts, complete execution logs or
feasibility proofs**. See the [instrumentation, identity, privacy and overhead
guide](https://marcosptproenca.github.io/rko-solver/decision-traces/) before enabling it or sharing event text.

## Optional private FastAPI/Docker service

Expose your existing decoder/objective and optional checker through an authenticated
HTTP API without rewriting the solver. The maintained reference includes a trusted
scheduling adapter, supervised job cancellation/timeouts, offline report downloads
and a resource-limited Docker Compose configuration. HTTP dependencies remain
optional; core optimization still requires only NumPy.

From the reviewed source checkout, build the SDK and generate a new service project.
Set `SDK_WHEEL` to the **actual absolute path** of the wheel just built in `dist/`;
do not assume an older PyPI release contains this API. The destination must not exist.
A working Docker Linux engine and Docker Compose are required.

```bash
uv sync --locked --extra service --extra reports
uv build --wheel
SDK_WHEEL=/absolute/path/to/the-reviewed-rko-wheel.whl
uv run --locked rko-solver-service scaffold ../my-private-service --wheel "$SDK_WHEEL"
cd ../my-private-service
export RKO_WHEEL="${SDK_WHEEL##*/}"
export RKO_SERVICE_API_KEY="$(uv run --no-project python -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker compose up --build -d
curl --fail http://127.0.0.1:8000/health
```

Submit bounded JSON to the reference adapter using the generated bearer secret:

```bash
curl --fail http://127.0.0.1:8000/jobs \
  -H "Authorization: Bearer $RKO_SERVICE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"instance":{"name":"demo","processing_times":[3,2,5],"due_dates":[3,6,9]},"options":{"algorithms":["SA","PSO"],"time_limit":0.2,"validate_solution":true,"include_solution":true,"report":true}}'
```

The response contains a job UUID; poll authenticated `GET /jobs/UUID` for its result.
Edit the generated `adapter.py` to bind your application's bounded schema and trusted
callbacks. Clients submit data/settings, **never Python code, imports or host paths**.

This is a **private single-tenant Linux-container reference**, not a public SaaS or
an untrusted-code sandbox. Keep the supplied localhost binding, authentication and
CPU/RAM/PID/disk limits. Only one job runs at a time; results expire and container
replacement can erase them. Native Windows supervision is unsupported; Windows/WSL
hosts use Linux Docker. Stop the reference with `docker compose down`.
See the [complete FastAPI/Docker guide](https://marcosptproenca.github.io/rko-solver/service/) for integration, API endpoints,
cancellation, retention, resource limits and deployment/security requirements.

## Run the supplied Knapsack adapter

```bash
uv run --locked --no-dev rko-solver src/rko_solver/examples/instances/kp50.txt \
  --limit-time 10 --execution process --random-state 42 \
  --output-csv results/kp50.csv
```

The supplied command handles Knapsack, not arbitrary input formats. Its two
bundled instances are correctness fixtures, not competitive-performance evidence.
See [terminal/configuration usage](https://marcosptproenca.github.io/rko-solver/running/) for custom instances,
TOMLs, progress output and worker limits.

## Documentation

Start at the [documentation index](https://marcosptproenca.github.io/rko-solver/): tutorials, how-to guides,
parameter reference and contributor guidance.

| Need | Guide |
| --- | --- |
| Solve your first problem | [Getting started](https://marcosptproenca.github.io/rko-solver/getting-started/) |
| Use any one algorithm without managing a pool | [Standalone algorithms](https://marcosptproenca.github.io/rko-solver/standalone/) |
| Load instance data and implement decoder/objective callbacks | [File-to-solution tutorial](https://marcosptproenca.github.io/rko-solver/problem-adapters/) |
| Understand fixed/adaptive modes and every search knob | [Parameter reference](https://marcosptproenca.github.io/rko-solver/parameters/) |
| Replace operators or implement an algorithm | [Extension guide](https://marcosptproenca.github.io/rko-solver/extending/) |
| Generate offline interactive HTML charts and data-only exports | [Analysis reports](https://marcosptproenca.github.io/rko-solver/reporting/) |
| Optionally check the final domain solution | [Post-solve validation](https://marcosptproenca.github.io/rko-solver/solution-validation/) |
| Optionally illustrate decoder logic without executing it | [Static decoder diagrams](https://marcosptproenca.github.io/rko-solver/decoder-diagrams/) |
| Record bounded author-emitted decisions from an original evaluation | [Observed decoder decisions](https://marcosptproenca.github.io/rko-solver/decision-traces/) |
| Connect trusted callbacks to a bounded private HTTP container | [Optional FastAPI/Docker service](https://marcosptproenca.github.io/rko-solver/service/) |
| Preview/build the development documentation website | [Documentation website](https://marcosptproenca.github.io/rko-solver/documentation-site/) |
| Configure the solver, callbacks, hyperparameters and results | [Usage and API guide](https://marcosptproenca.github.io/rko-solver/usage/) |
| Choose workers and understand startup/time accounting | [Execution modes](https://marcosptproenca.github.io/rko-solver/usage/#execution-modes), [workers](https://marcosptproenca.github.io/rko-solver/process-workers/), [startup](https://marcosptproenca.github.io/rko-solver/driver-initialization/) |
| Run from the terminal or avoid WSL setup overhead | [Running guide](https://marcosptproenca.github.io/rko-solver/running/) |
| Use the compiled engines and built-in problems | [Rust and C++ engines](https://marcosptproenca.github.io/rko-solver/native/) |
| Understand the C++ target and native-engine evidence | [Performance and native-engine evaluation](https://marcosptproenca.github.io/rko-solver/performance/) |
| Inspect measurements rather than speed claims | [Benchmarks](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md), [expanded quality study](https://marcosptproenca.github.io/rko-solver/prefill-expanded-work/) |
| Check algorithm differences and remaining risks | [C++ compatibility](https://marcosptproenca.github.io/rko-solver/cpp-compatibility/), [current limitations](https://marcosptproenca.github.io/rko-solver/status/) |
| See how the API, the three engines and the folders fit together | [Architecture](https://marcosptproenca.github.io/rko-solver/architecture/) |
| Develop, test or report an issue | [Contributing](https://github.com/MarcosPTProenca/rko-solver/blob/main/CONTRIBUTING.md) |

`engine` selects the Python, Rust or C++ engine (or `"auto"`), and `execution` selects
process, thread or sequential scheduling for the **Python** engine. Rust results versus pinned C++ and Python are reported with
their limits in [benchmarks](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md#native-engine-versus-c-and-python);
they do not extend to untested problem families, budgets or hardware. Prefill remains
opt-in: broader IPR benefits include regressions, and large-GRASP short-budget wins
can be startup-only.

## Project status and attribution

Version `0.3.0a3` is a prerelease, not a stable release. Matching or exceeding C++ RKO is a
performance target, **not an achieved claim**; read the [limitations](https://marcosptproenca.github.io/rko-solver/status/)
and [C++ compatibility](https://marcosptproenca.github.io/rko-solver/cpp-compatibility/) before relying on it.
Wheel/sdist installation and the full test suite are checked on Ubuntu, Windows and macOS;
that is not a supported-version or release guarantee.

The [LICENSE](https://github.com/MarcosPTProenca/rko-solver/blob/main/LICENSE) (Apache-2.0) applies.
[NOTICE](https://github.com/MarcosPTProenca/rko-solver/blob/main/NOTICE) lists the required attributions
and the fixture and vendored C++ licenses. New third-party additions and every publication
still require review and approval.
Report reproducible bugs through [GitHub issues](https://github.com/MarcosPTProenca/rko-solver/issues).
