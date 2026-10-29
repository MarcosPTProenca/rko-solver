# RKO

[![CI](https://github.com/MarcosPTProenca/rko-solver/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MarcosPTProenca/rko-solver/actions/workflows/ci.yml)

[Development documentation](https://marcosptproenca.github.io/rko-solver/)

A domain-neutral **random-key optimization** library with cooperating
metaheuristics, inspired by [RKO C++ v2](https://github.com/RKO-solver/RKO_Cpp_v2.0).
You provide a decoder and an objective to minimize; RKO searches for good key
vectors without assuming a problem format or solution representation.

> **Development documentation:** this README follows `main`, which contains changes newer
> than the published `0.3.0a4` prerelease. A matching version string does not mean matching code.
> Choose the correct [installation and documentation version](https://marcosptproenca.github.io/rko-solver/installation/).

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
  (Numba is an optional extra; see the [installation guide](https://marcosptproenca.github.io/rko-solver/installation/#optional-features)).
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

Keep the loaded data, decoder, objective and optional validation unchanged. Replace
only the solver block inside `main` with one of these forms.

Select one algorithm through the portfolio API:

<!-- doc-test: skip fragment replaces the solver block inside main; dedicated README tests execute it with loaded callbacks -->
```python
solver = RKO(
    decoder=decode, cost=cost, n_variables=data["lower"].size,
    algorithms=["SA"], adaptation="fixed", execution="sequential", random_state=42,
    hyperparameters={"SA": {"sa_max": 100, "t0": 1000.0, "alpha": 0.98}},
)
result = solver.solve(time_limit=1.0)
```

Or import `SA` from `rko_solver.metaheuristics` alongside the other imports and use
its direct solver class:

<!-- doc-test: skip fragment replaces the solver block inside main; dedicated README tests execute it with loaded callbacks -->
```python
solver = SA(
    decoder=decode, cost=cost, n_variables=data["lower"].size,
    adaptation="fixed", execution="sequential", random_state=42,
    hyperparameters={"SA": {"sa_max": 100, "t0": 1000.0, "alpha": 0.98}},
)
result = solver.solve(time_limit=1.0)
```

Both return the same `RKOResult` contract without requiring a user-managed pool or
search loop. See the [algorithm catalog and scalar settings](https://marcosptproenca.github.io/rko-solver/standalone/).

## Choose another workflow

Keep detailed contracts in the guides rather than duplicating them here:

| Need | Guide |
| --- | --- |
| Choose source versus a published package, or install optional features | [Installation](https://marcosptproenca.github.io/rko-solver/installation/) |
| Configure engines, execution, budgets, stopping and results | [Usage and API](https://marcosptproenca.github.io/rko-solver/usage/) |
| Load your own data and bind callbacks | [File-to-solution tutorial](https://marcosptproenca.github.io/rko-solver/problem-adapters/) |
| Choose one algorithm or tune fixed/adaptive settings | [Individual solvers](https://marcosptproenca.github.io/rko-solver/standalone/), [parameters](https://marcosptproenca.github.io/rko-solver/parameters/) |
| Use compiled or block objectives and automatic engine selection | [Rust and C++ engines](https://marcosptproenca.github.io/rko-solver/native/) |
| Check final solutions or generate offline reports | [Validation](https://marcosptproenca.github.io/rko-solver/solution-validation/), [reports](https://marcosptproenca.github.io/rko-solver/reporting/) |
| Inspect declared decoder structure or record original decisions | [Diagrams](https://marcosptproenca.github.io/rko-solver/decoder-diagrams/), [observed decisions](https://marcosptproenca.github.io/rko-solver/decision-traces/) |
| Convert a bounded mathematical model | [Model conversion](https://marcosptproenca.github.io/rko-solver/modeling/) |
| Host trusted callbacks privately or explore decoder evolution | [Service](https://marcosptproenca.github.io/rko-solver/service/), [experimental evolution](https://marcosptproenca.github.io/rko-solver/self-evolve/) |
| Extend the library or inspect evidence and limitations | [Architecture](https://marcosptproenca.github.io/rko-solver/architecture/), [contributing](https://github.com/MarcosPTProenca/rko-solver/blob/main/CONTRIBUTING.md), [status](https://marcosptproenca.github.io/rko-solver/status/) |

## Run repository examples

Examples live in the checkout's `examples/` directory. They are not included in
current development distributions, and the old installed `rko-solver` demo command
is removed. `rko-solver-service` remains available with the service extra.

Run the data-loading, solving and validation tutorial:

```bash
uv run --locked --no-dev python -m examples.instance_pipeline
```

For the supplied Knapsack instances, use the
[terminal guide](https://marcosptproenca.github.io/rko-solver/running/). Its instance format is
an example, not a required RKO input format or competitive-performance evidence.
Run all repository commands from the checkout root.

## Status and attribution

This is an alpha library, not a stable release or a universal C++ performance guarantee.
The [current limitations](https://marcosptproenca.github.io/rko-solver/status/) distinguish functional
checks, historical measurements and open qualification work. Read the
[identified benchmarks](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md)
instead of extrapolating from a language, one instance or one machine.

The [LICENSE](https://github.com/MarcosPTProenca/rko-solver/blob/main/LICENSE) and
[NOTICE](https://github.com/MarcosPTProenca/rko-solver/blob/main/NOTICE) retain project, fixture
and vendored-code attribution. Every publication still requires separate owner approval.
Report reproducible bugs through [GitHub issues](https://github.com/MarcosPTProenca/rko-solver/issues).
