# RKO

### Write a decoder. Not a metaheuristic.

[![CI](https://github.com/MarcosPTProenca/rko-solver/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MarcosPTProenca/rko-solver/actions/workflows/ci.yml)
[Development documentation](https://marcosptproenca.github.io/rko-solver/) ·
**Alpha** — see [current limitations](https://marcosptproenca.github.io/rko-solver/status/).

> This README follows `main`, which contains changes newer than the published
> `0.3.0a4` prerelease. A matching version string does not mean matching code.
> Choose the correct [installation and documentation version](https://marcosptproenca.github.io/rko-solver/installation/).

RKO is a domain-neutral **random-key optimization** library with cooperating
metaheuristics, inspired by [RKO C++ v2](https://github.com/RKO-solver/RKO_Cpp_v2.0).
Every solution is encoded as a vector of keys in `[0, 1)`; your decoder turns keys
into your domain's solution, your objective scores it, and the library's search
algorithms explore the key space without knowing anything about your problem. You
do not reimplement BRKGA, simulated annealing, VNS or a population loop for each
new problem — you implement two functions.

## What you implement

| Your code | Responsibility |
| --- | --- |
| `decoder(keys)` | Convert a key vector into your domain solution. |
| `objective(solution)` | Return a finite scalar to **minimize**. Passed as `cost`. |
| `read_data(source)` *(optional)* | Load your instance once, outside the search budget. |
| `validate(solution)` *(optional)* | Post-solve domain check, returns a `ValidationReport`. |

RKO calls only `decoder` and `objective` during search. Bind loaded data with
`functools.partial` or a small adapter class — neither is a required interface.
See the [complete data-loading tutorial](https://marcosptproenca.github.io/rko-solver/problem-adapters/).

## What it provides

- Eleven algorithms in one portfolio: BRKGA, BRKGA-CS, GA, GRASP, ILS, IPR, LNS,
  MultiStart, PSO, SA, VNS — sequential, thread or process execution, shared
  elites, owned RNG streams, immutable results.
- An optional compiled Rust engine (`engine="rust"`) measured **against a pinned
  C++ reference on Knapsack, TSP and Rastrigin: won 8 of 11 head-to-head rounds,
  tied 3, lost 0** ([full protocol and results](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md#native-engine-versus-c-and-python)).
  Every claim there is scoped to those problems, budgets and machine — read it
  before extrapolating.
- Python callbacks for any problem; **NumPy is the only required runtime
  dependency**. A prebuilt wheel needs no compiler; building from source needs
  Rust. No C++ toolchain or agent framework required.

## Install

Requires **Python 3.14+** and [uv](https://docs.astral.sh/uv/). Source installs
also need a [Rust toolchain](https://rustup.rs/) (the native extension builds
from source). CI tests Python 3.14 on Ubuntu, Windows and macOS (x86_64/ARM).

```bash
git clone https://github.com/MarcosPTProenca/rko-solver.git
cd rko-solver
uv sync --locked --no-dev
```

Run scripts with `uv run --locked --no-dev python your_script.py` from this
checkout. For reproducible use, record the Git revision alongside the lock file.

## First solve

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
    print("Solution:", decode(np.asarray(result.keys)))
    print("Objective:", result.cost)
```

Run with `uv run --locked --no-dev python first_solve.py`. `algorithms=["SA"]`
selects one algorithm through the portfolio API; swap in any
[other algorithm](https://marcosptproenca.github.io/rko-solver/standalone/#algorithm-catalog)
or add more for a cooperating portfolio, without touching `decode`/`objective`.
RKO's default execution is a **process portfolio**, which needs importable
callbacks and a guarded entry point; sequential execution above is the simplest
first run. See [execution modes](https://marcosptproenca.github.io/rko-solver/usage/#execution-modes)
before switching, and the
[file-backed tutorial](https://marcosptproenca.github.io/rko-solver/problem-adapters/)
for loading real instance data and optional post-solve validation.

Deadlines are cooperative — an already-running callback cannot be interrupted —
and `time_limit` gives **each selected algorithm** its own allowance under
sequential execution, not one allowance shared across the portfolio.

## Choose a workflow

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

Repository examples live in `examples/` (not installed with the wheel). Run
`uv run --locked --no-dev python -m examples.instance_pipeline` for the complete
read/decode/objective/validate tutorial, or see the
[terminal guide](https://marcosptproenca.github.io/rko-solver/running/) for the
supplied Knapsack instances.

## Status and attribution

This is an **alpha** library: not a stable API, not a universal performance
guarantee. The [current limitations](https://marcosptproenca.github.io/rko-solver/status/)
distinguish functional checks, historical measurements and open qualification
work. Read the [identified benchmarks](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md)
instead of extrapolating from one language, one instance or one machine.

The [LICENSE](https://github.com/MarcosPTProenca/rko-solver/blob/main/LICENSE) and
[NOTICE](https://github.com/MarcosPTProenca/rko-solver/blob/main/NOTICE) retain
project, fixture and vendored-code attribution. Every publication still requires
separate owner approval. Report reproducible bugs through
[GitHub issues](https://github.com/MarcosPTProenca/rko-solver/issues).
