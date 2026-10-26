# Getting started

[Documentation index](index.md) · [Standalone algorithms](standalone.md)

## Install

Requires Python 3.14+ and uv. From the repository root:

```bash
git clone https://github.com/MarcosPTProenca/rko-solver.git
cd rko-solver
uv sync --locked --no-dev
```

This is an unreleased development API; record the Git revision and lockfile.
The PyPI package named `rko` is a different project; this one is `rko-solver`.

## Run one algorithm

Save the following as `first_solve.py`. The decoder converts random keys into your
solution representation. The objective scores that representation; lower is better.
No elite pool, TOML edits or portfolio setup is required.

```python
import numpy as np
from rko_solver import RKO


def decode(keys: np.ndarray) -> np.ndarray:
    """Map unit-interval keys to a numerical solution in [-5, 5]."""
    return 10.0 * keys - 5.0


def objective(solution: np.ndarray) -> float:
    """Return the squared norm to minimize."""
    return float(np.dot(solution, solution))


if __name__ == "__main__":
    solver = RKO(
        decode, objective, n_variables=10, algorithms=["SA"],
        adaptation="fixed", execution="sequential", random_state=42,
        hyperparameters={"SA": {"sa_max": 100, "t0": 1000.0}},
    )
    result = solver.solve(time_limit=2.0)
    solution = decode(np.asarray(result.keys))
    print("Solution:", solution)
    print("Objective:", result.cost)
    print("Evaluations:", result.n_evaluations)
    print("Stopping reason:", result.stop_reason)
```

```bash
uv run --locked --no-dev python first_solve.py
```

This selects one algorithm through the main RKO API, with explicit sequential execution,
a private elite pool and an immutable `RKOResult`. To use the direct class API,
construct `SA(...)` from `rko_solver.metaheuristics` with the same settings but no
`algorithms` argument, then call `solver.solve(...)` unchanged. Select another
algorithm through RKO or another [public class](standalone.md#algorithm-catalog)
without changing your problem callbacks.
The example is a usage demonstration, not a recommended benchmark budget.

## Run a portfolio

Use the same `decode` and `objective` with the existing portfolio API:

```python
from rko_solver import RKO

solver = RKO(
    decode, objective, n_variables=10,
    algorithms=["SA", "PSO"],
    adaptation="fixed",
    execution="sequential",
    random_state=42,
)
result = solver.solve(time_limit=2.0)
```

Sequential execution gives **each** algorithm two seconds. The default process execution
has different accounting and requires importable callbacks and an entry-point
guard; read [execution modes](usage.md#execution-modes) before switching.

## Understand the result

- `keys`: immutable chromosome, not the decoded domain solution.
- `cost`: lowest finite objective observed; not a certificate of optimality.
- `elapsed_seconds`: end-to-end runtime; may exceed the requested search budget.
- `n_evaluations`: completed finite objective calls, including nested local search.
- `searches`: actual parameter snapshots, assigned seeds and stopping reasons.
- `adaptation`: requested mode; `searches[i].control` records effective adaptation.

Deadlines are cooperative: an already-running callback cannot be interrupted.
A seed controls library randomness, not callback RNGs or wall-clock scheduling.
The solver cannot infer feasibility from an arbitrary numeric objective. Your
adapter must enforce or penalize constraints and check the returned solution.

This numerical example needs no input file. For real instances, the application
calls `read_data` before solving and binds the loaded data to decoder/objective
methods. Follow the [complete file-backed scheduling tutorial](problem-adapters.md)
and then [understand parameter selection](parameters.md#fixed-versus-adaptive).
