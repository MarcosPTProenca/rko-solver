# Mathematical model conversion — development preview

`RKO.from_model` converts an existing supported LP/MILP model into owned generated
callbacks and runs the existing RKO algorithms. This remains a **bounded development
preview**, not a complete mathematical-programming solver. Existing handwritten
decoder/objective workflows are unchanged.

Current source has hosted functional and installed-extra checks for the pinned
frontends in the [integration workflow](https://github.com/MarcosPTProenca/rko-solver/blob/main/.github/workflows/ci.yml).
Read results for the exact reviewed revision, not as a permanent certification.
Historical performance measurements identify the preserved `13b6934` implementation,
not automatically the current source. Fresh scientific qualification remains separate.
See [version boundaries](installation.md) and [historical measurements](model-validation.md).

## Public workflow

```python
import importlib.util
from rko_solver import RKO

model = None
if importlib.util.find_spec("pyomo") is not None:
    import pyomo.environ as p

    model = p.ConcreteModel()
    model.x = p.Var(bounds=(0, 1))
    model.y = p.Var(bounds=(0, 1))
    model.balance = p.Constraint(expr=model.x + model.y == 1)
    model.objective = p.Objective(expr=model.x + 2 * model.y)
    solver = RKO.from_model(
        model, algorithms=["SA"], adaptation="fixed", engine="python",
        execution="sequential", repair="continuous", repair_max_sweeps=1024,
        random_state=42,
    )
    result = solver.solve(time_limit=0.1)
    solution = result.model.best_feasible
    if solution is not None:
        print(solution.direction, solution.objective)
        for assignment in solution.assignments:
            print(assignment.identifier, assignment.label, assignment.value)
    else:
        print(result.model.status)
else:
    print("Install rko-solver[pyomo] for this optional workflow.")
```

Generated models use `engine="python"`, with `execution="process"`, `"thread"`
or `"sequential"`. The `rust`, `cpp` and `auto` engines do not collect generated
model witnesses and are rejected by this factory, including changes after conversion. This
does not change ordinary callback execution or canonical `Numba`/`Blocks` problems.
The deprecated alternative constructors are removed from development `main`.

Here `model` is an already constructed, trusted native model from one qualified
frontend below. No handwritten decoder/objective is required. All eleven existing
algorithms remain available through normal selection; standalone callback-based
solver constructors retain their behavior.

Run the maintained equivalent mixed-integer example with any selected extra:

```bash
uv run --locked --no-dev --extra pyomo python -m examples.model_conversion pyomo
uv run --locked --no-dev --extra pulp python -m examples.model_conversion pulp
uv run --locked --no-dev --extra mathopt python -m examples.model_conversion mathopt
```

The example is an original reproducible formulation with an analytic maximum of
3.5. Search may return a lower feasible objective; it does not certify optimality.

### Search incumbent versus model solution

`result.keys` and `result.cost` are always the actual **search-score incumbent**.
`result.model.search_incumbent` identifies that same completed invocation, with its
original objective and feasibility diagnostics. In weighted mode it may be infeasible.
`result.model.best_feasible` is the independently retained best feasible original
objective, with its own keys, score, immutable assignments and row diagnostics.
It can differ from the root incumbent and can survive algorithm rejection/pool eviction.

`result.model.status` is `feasible` or `no_feasible_found`. In the latter case
`best_feasible` is `None`; an infeasible objective appears only in search-incumbent
diagnostics, never as a primary solution. Neither status claims mathematical
infeasibility, optimality, an optimality gap or an external solver certificate.

Explicit portable export uses `result.model.to_dict()`. Normal
`OptimizationReport` JSON/HTML excludes assignment coordinates/keys, labels histories
as search scores, and presents model status/original objective separately. Optional
`assess_result` still assesses the root search incumbent, **not** another feasible
witness; model assessments explicitly identify this scope. Model results already
contain their completed validation evidence and do not need post-search replay.

Conversion is timed separately in `result.model.compilation_seconds`; it is not
included in solve work. `result.model.time_to_first_feasible_seconds` measures the
first completed finite feasible evaluation since `solve` began, including any
execution/startup work but excluding prior conversion. Missing/unknown evidence
has `None`, not a fabricated zero. Fully fixed/constant snapshots are evaluated once directly:
empty keys, execution `direct`, one counted evaluation, zero workers and empty algorithm,
search and initialization metadata. No dummy key, pool or fictitious search is created.

### Explicit trusted construction

`RKO.from_model(native, construction=trusted_callable, ...)` opts into custom
construction while retaining original model validation/scoring. The callable has
signature `decode(keys, instance)` and returns complete original-order numerical
values. `instance` is immutable original mathematics and effective domains, not an
evaluator/native object; fixed coordinates consume no key. This does not load
artifact paths or execute generated source on the host. See the experimental
[Self-Evolve model bridge](self-evolve.md#experimental-mathematical-construction-bridge-m6)
for externally isolated program artifacts.

The ordinary numerical scorer, tolerances and configured continuous restoration
remain authoritative. Completed search incumbents and best feasible original
objectives are retained without rerunning construction/objective. Constructed-model
results expose completed evaluation/infeasibility counts and capped distinct output
identities through optional `statistics`; missing diagnostics are not invented.
Ordinary model/callback defaults and default result JSON remain unchanged.
An explicit `construction_deadline` is an optional finite absolute
`time.perf_counter()` ceiling for construction-backed search/initialization
sessions; it does not reset after solver setup. It is not exposed to the
constructor's immutable mathematical context. The existing per-session
first-completion grace remains; partial later invocations are not retained. Custom
construction supports portable assignments/native-ID mapping but not transactional
native write-back; `write_back` raises `NotImplementedError`. Existing default
native write-back remains available. Fully fixed models bypass construction/search.

## Implemented foundations

The shared numerical layer supports bounded continuous, integer and binary
variables, one linear minimization/maximization objective with a constant, and
linear equalities, inequalities and ranged rows. Fixed variables need no keys.
It owns its data and can be serialized without retaining a native modeling object.

Thin extractors have local Ubuntu/WSL Python 3.14 tests against these exact releases:

| Development extra | Qualified API candidate |
| --- | --- |
| `pyomo` | Pyomo 6.10.1 |
| `pulp` | PuLP 4.2.0 |
| `mathopt` | OR-Tools MathOpt 9.15.6755 |

The default installation still requires only NumPy. Frontend imports are lazy;
using one frontend does not import the others. Extras pin the tested releases;
wider version compatibility is not claimed. Ubuntu/Windows Python 3.14 CI covers
each extra separately and all three together, but a configured job is not evidence
that its hosted qualification has passed.

To exercise the development integrations from this checkout:

```bash
uv run --extra pyomo --extra pulp --extra mathopt pytest tests/rko/test_model_adapters.py -q
```

No external LP/MIP optimizer is called by extraction, decoding or these tests.
Native runtimes required by the modeling libraries remain optional dependencies.

## Conversion boundary

- Pyomo respects active block/objective/constraint semantics. It rejects multiple
  active objectives, nonlinear expressions, GDP/logical/SOS components, open or
  noninterval domains, and populated export suffixes whose solver-specific semantics
  cannot be preserved. Closed continuous intervals and contiguous integer domains
  are supported. Fixed expressions retain source-structure evidence for drift checks.
- PuLP uses its 4.x public variable/constraint IDs and expression API, not names or
  `toDict()` for mathematical extraction. In 4.2.0, `toDict()` omits the objective
  constant. Constraint bounds retain the native RHS (the negative of PuLP's
  constraint `constant`), rather than shifting every row to a zero bound and
  accidentally disabling its relative allowance. Duplicate display names are
  permitted because IDs remain distinct. SOS groups are rejected. Legacy PuLP 3.x is not supported by this adapter.
- MathOpt audits every populated model-proto feature and primary objective field.
  Quadratics, auxiliary objectives, indicators, cones, SOS and other unsupported
  fields fail explicitly. This does not add CP-SAT or MPSolver support. IDs, rather
  than potentially duplicate names, identify variables and rows.

Every free variable requires finite effective bounds. Explicit search bounds can
restrict/intersect the original domain but cannot relax it. Bounds implied only
by coupled constraints are not derived. Unsafe numeric data fail preflight;
there is no general presolve, solver-assisted repair or silent linearization.

Snapshots and source fingerprints are separate. Search-bound overrides and
scoring/tolerance settings do not change the source fingerprint. Supply overrides as
`search_bounds={stable_variable_id: (lower, upper)}`; `None` leaves that side unchanged.
IDs use `pyomo:<ComponentUID>`, `pulp:<native id>` or `mathopt:<native id>`; inspect
`ModelAdapter.extract(model).variables` from `rko_solver.modeling.adapters` before compiling
if you need the model's original stable ordering. Native solution
guesses are not mathematical source data; changing bounds, coefficients, fixed
values, objectives or supported source structure invalidates reconstruction.

## Numerical and feasibility rules

Continuous keys map affinely. Integer mapping is
`lower + floor(key * (upper - lower + 1))`, clipped at the upper endpoint: a
specified half-up interpretation of the paper-inspired bounded mapping, not
Python truncation. Keys zero and one are valid.

Each finite row side `b` has allowance
`absolute_tolerance + relative_tolerance * abs(b)`, defaulting to `1e-6` and zero.
Both sides are checked independently. Nonzero relative allowances refer to the
extracted frontend's finite row bounds; algebraic shifts/scaling of a row can change
those allowances. Default relative zero deliberately avoids implicit RHS-scaled
feasibility margins. Diagnostics distinguish raw residuals from tolerance-adjusted violations. Bounds and integrality are enforced by mapping,
not relaxed by these row allowances.

Feasibility-first scoring places feasible candidates in `[0, 0.5]` and infeasible
ones at or above one. Float normalization can tie distinct objectives; independent
completed-evaluation retention compares the original objectives directly. Weighted
penalty mode requires an explicit positive finite weight and squares raw residuals
only when their side exceeds its tolerance allowance, in original row units:

```python
chosen_positive_weight = 1000.0
if model is not None:
    solver = RKO.from_model(
        model, scoring="weighted_penalty", penalty_weight=chosen_positive_weight,
        engine="python", execution="sequential",
    )
```

The caller supplies `chosen_positive_weight`; there is no universal recommended
weight. An infeasible score incumbent can be cheaper than every feasible candidate,
which is why the original-objective feasible archive is independent.

The conservative numerical envelope caps magnitudes, accumulated expressions and
worst-case scores at `1e100`, with effective integer endpoints/interval width at
most `2**52`. Integer scalar data that would be rounded by float64 conversion
are rejected before coercion, including exact Pyomo integer bounds/constants and
coefficients. This prevents a narrow continuous interval at a huge integer offset
from silently becoming a fixed point. Already floating-point native data retain
the frontend's own representation. Runtime finite checks remain. These limits
reject unsafe models; they do not provide arbitrary-precision arithmetic.

The raw box map does **not** construct coupled feasibility. The public factory
therefore defaults to `repair="continuous"`: bounded cyclic projections jointly
restore movable continuous coordinates against all affine rows and their box.
Integer, binary, fixed and singleton values remain exactly as mapped. Raw feasible
points return unchanged; original tolerances and objective semantics do not change.

The common default work cap is **1,024 full sweeps**, configurable from 1 to
10,000. The initially proposed cap256 failed strict transportation controls and
was explicitly revised by the owner before qualification. There is no per-case
policy or hidden tolerance increase. No dense matrix, optimizer or new runtime
dependency is used. Runtime/storage remain sparse: `O(S*(n+nnz+m))` work and
`O(n+nnz+m)` space for capS.

`repair="none"` explicitly retains the original box-only mapping. Low-level
`CompiledLinearModel`/source compilation also default to none for compatibility;
the public factory passes its selected policy explicitly. Fully fixed and
all-integer models bypass continuous restoration.

Each retained solution exposes `solution.repair`: policy, outcome and completed
sweeps. Outcomes include `disabled`, `already_feasible`, `repaired`, `work_limit`,
`no_continuous_variables` and `fixed_obstruction`. A work-limited point is scored
honestly and may be infeasible. Repair never rounds relaxed integer variables or
certifies that a different integer assignment has no feasible completion.

Completed mappings are deterministic for fixed keys/policy. Cooperative owner
cancellation/deadlines abort a partial invocation rather than publish a
clock-dependent assignment; partial repairs do not increment evaluations or
update feasible evidence. The existing first-evaluation grace is retained, so
one first bounded callback may exceed a tiny nominal allowance. Native model
objects and budget callbacks are never stored in the portable numerical snapshot.

A failure to find feasibility is not a mathematical infeasibility certificate.
No solution is labelled optimal and no optimality gap is certified.

### Historical equality diagnosis and restoration boundary

Before restoration, a deterministic control reproduced the limitation even for `x + y = 1` on the
unit square: `[0.25, 0.75]` is accepted, while none of 2,048 independently sampled
key pairs from seed 2026 is feasible at absolute tolerance `1e-6`. Keys are mapped
independently; the second coordinate is **not** reconstructed as `1 - x`.
Equality-respecting pairs `[u, 1-u]` are accepted in all 2,048 controls. This is a
candidate-space problem, not evidence of a bad equality evaluator.

For the exact-demand transportation fixture, all three frontends accepted an
independently checked feasible assignment with objective 10,100. For 2,048 fixed
random candidates per frontend, original-data and generated checks agreed in
all cases; maximum activity disagreement was about `3.64e-12`, below the declared
`1e-6` allowance. The archive retained the feasible diagnostic control. Real
instrumented BRKGA searches also produced no feasible completion in the tested
three seeds at either 0.1 or 1 second; no evaluated feasible witness was lost.

An offline orthogonal projection is **not** a sufficient repair: it satisfied
all five demand balances in 2,048 diagnostic points, but only five also respected
variable bounds and none satisfied all remaining constraints. Clipping to bounds
broke most balances again. These points were never injected into RKO search.
The implemented restoration revisits all row sets and the box, rather than
using that one-shot projection/clip shortcut. Integer choices are frozen; neither
transport-specific repair nor looser hidden tolerances is used.

Finite squared penalties also do not force equality feasibility. With objective
`x + 2y`, `x+y=1`, and weight 1,000, feasible `[1,0]` scores `1`, but infeasible
`[0.9995,0]` scores `0.99975` despite a residual of `0.0005`. This is the explicit
weighted-mode tradeoff, not a finite-value or archive bug. Default feasibility-first
correctly prefers the feasible point but still needs a way to generate one.
Restoration must be independently qualified for the model classes and budgets
being claimed. It is not a finite-time feasibility guarantee for every LP or a
constructive integer solver. Explicit box-only weighted mode retains this tradeoff.

## Explicit native assignment application

`result.model.native_mapping(model)` reconstructs native-variable mappings against
an explicitly supplied unchanged compatible model. Portable source records, numerical
callbacks and completed witnesses contain no native model references; spawned searches
receive owned numerical data only. `result.model.write_back(model)` explicitly applies
only `best_feasible`, never an infeasible root incumbent. A result does not retain or
implicitly mutate the source object. Reconvert after source edits.

Native write-back never runs automatically during search. It revalidates source
identity, the snapshot and the feasible witness before mutation. Assignment
validation is separate from optimization counts/timing, not a replay used to
recover a missing witness.

- **Pyomo:** set stored variable values, without fixing variables or altering bounds.
- **PuLP 4.2.0:** transactional write-back requires **all prior stored values to be
  initialized and finite**. Its public `varValue = None` setter is a no-op, so an
  unset value cannot be restored on failure. Fresh/uninitialized problems are
  rejected before any mutation; use explicit assignment export instead. This is
  a native API limitation, not permission to leave partial assignments behind.
- **MathOpt:** native write-back is explicitly unsupported because variables have
  no assignment-value slot. Export assignments/native mappings instead; bounds are
  never modified to simulate a solution and no native `SolveResult` is fabricated.

Setter failures trigger restoration of all attempted targets. Silent restoration
failures are checked and any rollback failure is reported explicitly.

## Remaining release gates

The [practical model validation](model-validation.md) records 480 repaired runs
and a separate 480-run same-source box control, independently checked across all
three frontends. Exact-demand transportation found feasibility in60/60 repaired
runs versus0/60 box runs at unchanged tolerances. Objective quality and repair
cost remain model-dependent; this is not industrial certification.

Local Linux/native-Windows tests, installed wheel/sdist smokes, restoration costs,
archive/IPC measurements and ordinary-callback controls are recorded. Scheduling
and Knapsack protection intervals remain inconclusive, not passed or confirmed
regressions. Hosted CI, larger application/owner-sanitized model acceptance and
release approval remain outstanding. See [current limitations](status.md).
Do not treat local integrations or published frontend wheels as proof of full release
qualification, general feasibility guarantees or superiority over dedicated solvers.
