# Read instance data and implement a problem adapter

[Documentation index](index.md) · [Getting started](getting-started.md)

## Your three problem-specific functions

You implement **only the reader, decoder and objective** for your problem:

1. `read_data(source)` loads the instance; omit it when your data is already in memory.
2. `decoder(keys)` constructs your domain solution using the loaded data.
3. `objective(solution)` returns a finite scalar to minimize, using the loaded data.
   Pass this callable as the library's `cost` argument.

The library supplies the metaheuristics, pool and default search operators. Selecting
an algorithm and setting its dimension/budget are configuration, not additional
implementations. This guide uses a class to hold data conveniently, **not** because
a class or a fourth problem interface is required. Functions with bound data work too.

## Where does `read_data` go?

**Read your instance once before calling the solver.** Keep the resulting data in
an importable adapter whose decoder and objective methods can access it:

```text
instance file / in-memory source
    -> read_data(source)
    -> validated instance
    -> problem adapter holding that instance
    -> decode(keys) -> domain solution -> cost(solution)
    -> any standalone metaheuristic OR an RKO portfolio
    -> best keys -> explicit final decoding -> optional report
```

The optimizer searches key vectors; your callbacks tell it what those keys mean
for the **loaded instance**. They do not have to operate without data. RKO neither
infers your file format nor calls a special reader inside each metaheuristic.
There is no required environment superclass or new pipeline framework.

The numerical quickstart uses a sphere objective and legitimately needs no file.
The following file-backed scheduling example shows the missing data lifecycle.

## A complete file-backed example

The maintained implementation, which lives in the repository's `examples/` directory and is not
installed with the wheel (run these snippets from the repository root), is
[`examples.instance_pipeline`](https://github.com/MarcosPTProenca/rko-solver/blob/main/examples/instance_pipeline.py).
It solves an illustrative single-machine scheduling problem: each job has a
processing duration and due date. The decoder constructs completion times from
priority keys; the objective minimizes total tardiness.

### 1. Define your input format

Save this as `my_instance.json`:

```json
{
  "name": "four-job scheduling tutorial",
  "processing_times": [3, 2, 5, 1],
  "due_dates": [4, 6, 7, 3]
}
```

Every job needs a positive integer processing time and a nonnegative integer due
date in the same units. Both lists must be nonempty and have equal length; booleans
are not accepted as integers. The reader rejects missing/extra fields and an empty
name. This schema is the **example application's format**, not a format required
by the library.

### 2. Load data and bind the callbacks

The example's `read_data(path)` returns a `SchedulingInstance` with immutable tuples.
`SchedulingProblem(instance)` stores it. Its two callbacks use that same snapshot:

| Method | What it does with the instance |
| --- | --- |
| `problem.n_variables` | Derives one key per job from the loaded processing times |
| `problem.decode(keys)` | Stably sorts job indices by key, then accumulates the **loaded processing times** to construct a schedule |
| `problem.cost(solution)` | Sums `max(0, completion_time - loaded_due_date)` for scheduled jobs |

For keys `[0.2, 0.9, 0.7, 0.1]`, jobs run in order `3, 0, 2, 1`. Their completion
times are `1, 4, 9, 11`; total tardiness is `0 + 0 + 2 + 5 = 7`.
Changing processing times changes the decoded schedule; changing due dates changes
its objective. Neither callback reopens the file.

### 3. Run one algorithm

Save the following as `solve_schedule.py` beside your JSON file:

```python
import numpy as np
from pathlib import Path
from examples.instance_pipeline import SchedulingProblem
from examples.instance_pipeline import read_data
from rko_solver.metaheuristics import SA


def main() -> None:
    """Load one application instance and optimize its scheduling priorities.

    Raises:
        OSError: If the input file cannot be read.
        ValueError: If the instance or search settings are invalid.
    """
    instance = read_data(Path("my_instance.json"))
    problem = SchedulingProblem(instance)
    solver = SA(
        problem.decode, problem.cost, n_variables=problem.n_variables,
        adaptation="fixed", execution="sequential", random_state=42,
        hyperparameters={"SA": {"sa_max": 100, "t0": 1000.0}},
    )
    result = solver.solve(time_limit=2.0)
    solution = problem.decode(np.asarray(result.keys))
    print("Schedule (job index, completion time):", solution)
    print("Total tardiness:", result.cost)


if __name__ == "__main__":
    main()
```

From the source checkout:

```bash
uv run --locked python solve_schedule.py
```

Substitute `PSO`, `GRASP`, `GeneticAlgorithm` or any other
[built-in class](standalone.md#algorithm-catalog) without changing the reader or
adapter. All public classes use `solver.solve(...)` with the same lifecycle as RKO;
this example explicitly requests sequential execution. These classes are
metaheuristics, not trained models that need their own input-file loader.

### 4. Use the same instance in RKO

To use a portfolio, replace only the solver call in `main`:

<!-- doc-test: skip continues the program above, whose names are local to main() -->
```python
from rko_solver import RKO

solver = RKO(
    problem.decode, problem.cost, problem.n_variables,
    algorithms=["SA", "PSO"], adaptation="fixed",
    execution="process", n_jobs=2, random_state=42,
    hyperparameters={"SA": {"sa_max": 100}, "PSO": {"p_size": 20}},
)
result = solver.solve(time_limit=2.0)
```

Keep `read_data`, adapter construction and `solve` inside the guarded main function.
The adapter class itself is module-level/importable, not defined inside `main`.
Processes receive serialized copies of its data; the source file is not reopened
on every evaluation. See [execution modes](usage.md#execution-modes): sequential mode
gives each selected algorithm its own allowance, unlike the shared process deadline.

### 5. Run the supplied pipeline directly

The synthetic fixture is packaged, so no current-working-directory input is needed:

```bash
uv run --locked python -m examples.instance_pipeline
uv run --locked python -m examples.instance_pipeline my_instance.json \
  --algorithms SA PSO --execution process --n-jobs 2 --time-limit 2
```

Use `--execution thread` or `sequential` for those modes; `--n-jobs` is process-only.
All canonical algorithm names supported by RKO can be selected with `--algorithms`.
The short default budget is a usability demonstration, not performance evidence.

To generate a report with the existing optional reports feature:

```bash
uv sync --locked --extra reports
uv run --locked --extra reports python -m examples.instance_pipeline \
  my_instance.json --time-limit 2 --report schedule.html
```

The report's parent directory must already exist. History is captured only when
this example requests a report. Reports do not reload inputs, reevaluate objectives
or claim that the solution is optimal. See [reporting](reporting.md).

## Reuse the pattern for another problem

The library only needs these callbacks:

```text
read_data(source) -> your instance
YourProblem(instance).decode(keys) -> your solution
YourProblem(instance).cost(solution) -> finite scalar to minimize
```

For a different domain, replace the JSON reader/schema, adapter data and the two
methods—not algorithm internals. Data can come from files, a database, an API or
an existing in-memory object; load/validate it in your application before solving.

The existing [Knapsack runner](https://github.com/MarcosPTProenca/rko-solver/blob/main/examples/run_kp_cpp_instances.py) is an
unrelated example: `read_kp_instance` loads weights, benefits and capacity;
`KnapsackCost(instance)` stores them for scoring. Its threshold decoder does not
need the data itself, unlike scheduling construction. Knapsack input and execution
are documented in [the running guide](running.md). Do not treat either adapter as
a universal input format or benchmark of generic-engine performance.

### Functions and partials are also supported

A class is convenient for related callback state, not mandatory. Module-level
functions bound with `functools.partial` also work. With the concrete adapter above,
these are explicit equivalents of its bound methods:

<!-- doc-test: skip continues the program above, whose names are local to main() -->
```python
from functools import partial

decode = partial(SchedulingProblem.decode, problem)
cost = partial(SchedulingProblem.cost, problem)
```

Pass those callables where you previously passed `problem.decode` and
`problem.cost`. A lambda/nested closure can work in serial exploration but is not
a safe spawn-compatible script pattern. Avoid global instance variables and
module-import side effects. A database connection or open file handle is not a
replacement for a serializable data snapshot.

## Data, timing and memory ownership

- Parsing/preparation is outside `solve` and therefore outside its requested budget
  and reported invocation time. Final decoding/reporting is also separate work.
- Read once in the parent does **not** mean one RAM copy across process workers.
  Large data is serialized to workers and can multiply memory use. Limit `n_jobs`
  explicitly on constrained machines such as WSL.
- This example uses tuples of integers. A frozen dataclass containing mutable NumPy
  arrays/lists does not make those contents immutable; manage mutation yourself.
- Thread callbacks can share application state. Keep it read-only or synchronize
  explicitly. Do not mutate instance data during a solve or rely on reloading it.
- Memory mapping/shared-memory designs for large instances are advanced explicit
  choices, not automatic properties of an ordinary adapter.

## Representation, constraints and final decoding

| Domain representation | Typical decoder | Caution |
| --- | --- | --- |
| Permutation | `np.argsort(keys, kind="stable")` | Define deterministic ties and apply loaded domain data |
| Binary decisions | `keys > 0.5` | Thresholding does not enforce capacity |
| Bounded variables | `lower + (upper - lower) * keys` | Bounds belong to the validated instance |
| Structured assignment | Rank keys and construct admissible choices | Decoder work counts toward search time |

For maximization, minimize `-profit(solution)`. Negating the reported objective is
not the raw profit when it also includes penalties. The objective must be finite,
but a finite penalty is **not** a domain-feasibility certificate.

Reader checks establish input shape/domains before any search worker starts;
existing solver checks handle settings and finite costs. Domain validation of a
final solution is an application responsibility. You can opt into
[`assess_result`](solution-validation.md) after solving; the scheduling adapter
supplies an optional `validate` method, but no checker is required to optimize.
Use `--validate-solution` in the example CLI to enable it. Independently inspect
your returned solution and report feasibility separately when appropriate.

`result.keys` contains coordinates, not the decoded application solution. Final
`problem.decode(np.asarray(result.keys))` is an explicit post-run reconstruction.
This example's decoder is deterministic, so the reconstruction matches its recorded
objective. A stochastic/stateful decoder or changed instance can produce a different
solution; keys alone are not an original execution witness. Do not silently call
the objective again and add that work to the search evaluation count.

Use owned callback RNGs if needed; `random_state` controls library randomness,
not your callback generators. Raise informative parser/callback exceptions rather
than hiding bad data behind a dummy cost. Deadlines are checked between callbacks;
a running callback itself cannot be interrupted.
