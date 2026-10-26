# Optional final-solution validation

[Documentation index](index.md) · [Problem data pipeline](problem-adapters.md)

Optimization still requires only your decoder and objective, plus a reader when
loading external instances. A **validator is optional** application code. It checks
the reconstructed final solution after optimization, not every candidate during search.

A finite cost, successful run, passed parameter validation or `stop_reason=completed`
is not a certificate of domain feasibility or optimality.

## Use a checker after solving

The scheduling adapter from the [data-loading tutorial](problem-adapters.md) includes
an optional checker. This complete script uses its packaged synthetic fixture:

```python
from pathlib import Path
from rko_solver import assess_result
from rko_solver.examples import instance_pipeline
from rko_solver.examples.instance_pipeline import SchedulingProblem
from rko_solver.examples.instance_pipeline import read_data
from rko_solver.metaheuristics import SA


def main() -> None:
    """Optimize a loaded instance and explicitly assess its final reconstruction.

    Raises:
        OSError: If the input cannot be read.
        ValueError: If instance/search settings are invalid.
    """
    path = Path(instance_pipeline.__file__).with_name("instances") / "scheduling.json"
    problem = SchedulingProblem(read_data(path))
    solver = SA(
        problem.decode, problem.cost, problem.n_variables,
        adaptation="fixed", execution="sequential", random_state=42,
    )
    result = solver.solve(time_limit=2.0)
    assessment = assess_result(
        result, decoder=problem.decode, validator=problem.validate,
    )
    print("Objective:", result.cost)
    print("Check status:", assessment.validation.status)
    print("Declared feasibility:", assessment.validation.feasible)
    print("Decoded solution:", assessment.solution)


if __name__ == "__main__":
    main()
```

`assess_result` decodes a detached NumPy copy of the final keys **once**, calls the
checker **once** when supplied, and returns a `SolutionAssessment`. It has no
objective argument and does not implicitly reevaluate cost. Reader/data binding
and solver settings are unchanged. The same helper works after standalone classes
or RKO portfolios using sequential, thread or process execution; assessment runs in
the caller after those searches have finished.

Do not decode again just to get the payload: use `assessment.solution`.

## Implement your own optional checker

Any callable can accept the decoded payload and return a `ValidationReport`.
Use a function, partial or adapter method; no validator base class is required.
The checker should independently compute constraints from the loaded instance,
not merely return whether the optimization objective happened to be finite.

For example, a capacity check can return this report from your checker:

```python
from rko_solver import ValidationCheck
from rko_solver import ValidationReport

check = ValidationCheck(
    name="capacity",
    status="passed" if total_weight <= capacity else "failed",
    message="Compare total selected weight with the loaded capacity.",
    value=total_weight,
    limit=capacity,
    unit="kg",
)
validation = ValidationReport(
    checks=(check,),
    feasible=bool(total_weight <= capacity),
    checker="my_capacity_checker",
    checker_version="1",
)
```

Here `total_weight` and `capacity` are values computed by **your** checker. If other
hard constraints exist, check them too before declaring full domain feasibility.
A passed subset is not proof of complete coverage. Bare booleans/dictionaries are
not checker outputs: return a structured `ValidationReport` for unambiguous status.

The supplied scheduling checker verifies one integer pair per job, every job
exactly once, and start-at-zero/no-idle cumulative processing times. It recomputes
those constraints without calling its decoder or objective. Due dates are **soft
objective terms**: tardiness can be positive while the schedule is feasible.
Malformed structure prevents dependent checks, which remain `not_checked`.
This is an example domain checker, not a universal validator.

## Status and feasibility are different fields

| Outcome | Aggregate `status` | `feasible` |
| --- | --- | --- |
| No checks / omitted checker | `not_checked` | `None` |
| Some checks not run, no failures | `not_checked` | Normally `None` |
| Every implemented check passed | `passed` | Only what the checker explicitly declares, otherwise `None` |
| A check failed or checker declares infeasible | `failed` | Explicit `False` or undeclared `None` |

Failure takes priority over missing checks. Declaring `feasible=True` requires at
least one check and all supplied checks passed; contradictory/incomplete reports
raise `ValueError`. The library still cannot know whether your supplied checks
cover all the domain constraints. It never derives feasibility from numerical cost.

A failed check is a completed diagnostic result. It **does not** repair the solution,
change `result.cost`, rerank past candidates, recover the best feasible incumbent,
or turn the run into an infrastructure error. A decoder/checker bug raises its
original exception; the successful numerical result is not cleared or overwritten.

## Timing, randomness and reconstruction

- `assessment.elapsed_seconds` includes final decoding and checking, separately from
  `result.elapsed_seconds`, evaluation counts and search/initialization budgets.
- `assessment.origin` is always `post_run_decode`. A stateful/stochastic decoder
  or changed instance may produce something different from the original evaluation.
  The assessment validates this **reconstruction**, not an original execution witness.
- Objective consistency is **not checked** automatically. If needed, explicitly
  recompute it outside search with appropriate deterministic assumptions/tolerances.
- Callbacks manage their own randomness and side effects. The helper does not seed,
  restore or otherwise manage RNGs used inside your decoder/checker.
- Checkers are expected to inspect payloads read-only. The assessment record is
  frozen, but arbitrary domain objects are not recursively copied/frozen.
- Decoder/checker callbacks are trusted application code and are not sandboxed or
  forcibly interrupted by this helper. Keep them bounded.

## Include diagnostics in HTML and JSON

```python
from rko_solver.reporting import OptimizationReport

report = OptimizationReport(result, assessment=assessment)
report.write_html("checked-run.html")
report.write_json("checked-run.json")
```

The report adds a validation section with each check's status/message/value/limit,
checker identity, declared feasibility, reconstruction origin and separate time.
Without an assessment it explicitly reports `not_checked`, not success.
Only an assessment associated with the same numerical result is accepted.
Existing `metrics=` values remain user-provided metrics, not promoted to checks.

`assessment.to_dict()` exports diagnostics, origin and duration only. It excludes
`assessment.solution`, the source `result`, keys and arbitrary domain payloads.
Use this method for data-only export; generic dataclass serialization would include
its in-memory fields and can expose payloads. The HTML/JSON report uses the safe
export. Text is escaped but messages/measurements may still contain business-sensitive
values you intentionally supplied. Review them before sharing.

Diagnostic bounds: at most 1000 uniquely named checks; names 128 characters,
messages 2048, unit labels 64, scalar string values/limits 256. Checker identity
and version are limited to 256 and 128 characters. Measurements/limits must be
finite Python primitive scalars (or None), not arrays/dictionaries/payloads.
Convert NumPy scalar values explicitly to the appropriate Python float/int/bool;
do not truncate fractional measurements just to fit an integer type.

Run the packaged example with assessment enabled explicitly:

```bash
uv run --locked python -m rko_solver.examples.instance_pipeline --validate-solution
uv run --locked --extra reports python -m rko_solver.examples.instance_pipeline \
  --validate-solution --report checked-schedule.html
```

HTML requires the optional reports extra; validation and JSON do not require Plotly.
The report output directory must exist. A rendering error does not erase the
successful numerical result or the in-memory assessment.

## Not part of this feature

Per-candidate feasibility filtering, repair, early stop, best-feasible archives,
objective consistency proofs, optimality certification and original decision-witness
capture. Those change search semantics or require different data and remain separate.
