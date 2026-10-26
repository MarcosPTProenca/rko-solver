# Individual algorithm solvers

[Documentation index](index.md) · [Parameter reference](parameters.md)

All eleven public algorithm classes use the **same object lifecycle as RKO**:
configure a solver with decoder/objective/data bindings, then call `solver.solve`.
There is no separate class-level optimization call or manual pool requirement.

```python
from rko_solver.metaheuristics import PSO

solver = PSO(
    decoder=decode,
    cost=objective,
    n_variables=64,
    adaptation="fixed",
    execution="sequential",
    random_state=42,
    hyperparameters={"PSO": {"p_size": 20, "w": 0.6}},
)
result = solver.solve(time_limit=60.0)
print(result.cost, result.keys)
print(solver.result_ is result)
```

Use callbacks from [getting started](getting-started.md) or your own
[file-backed problem](problem-adapters.md). The same decoder, objective and optional
post-solve validator work without changes for a portfolio or an individual solver.

## Exactly the same execution pipeline

The individual solvers inherit `RKO.solve` itself, not a copied algorithm runner.
The only selection difference is that each class fixes one canonical algorithm.
The equivalent portfolio-interface construction is:

```python
from rko_solver import RKO

solver = RKO(
    decoder=decode, cost=objective, n_variables=64,
    algorithms=["PSO"], adaptation="fixed", execution="sequential", random_state=42,
    hyperparameters={"PSO": {"p_size": 20, "w": 0.6}},
)
result = solver.solve(time_limit=60.0)
```

Both build fresh state per solve, create the pool, use owned library RNGs, protect
callback boundaries, enforce cooperative search budgets and return immutable
`RKOResult` metadata. Both expose `result_` and clear it before a failed attempt.
You can retain a configured solver and call `.solve(...)` again. Application
callback state is still yours to manage; it is not reset automatically.

## Algorithm catalog

| Class imported from `rko_solver.metaheuristics` | Fixed algorithm / hyperparameter key | Search strategy | Adaptive parameters? |
| --- | --- | --- | --- |
| `SA` | `SA` | Thermal acceptance, cooling and local refinement | Yes |
| `ILS` | `ILS` | Perturb/refine the incumbent, retain improvements | Yes |
| `VNS` | `VNS` | Increase perturbation neighborhoods until improvement | Yes |
| `LNS` | `LNS` | Destroy/repair keys and apply thermal acceptance | Yes |
| `BRKGA` | `BRKGA` | Elite-biased crossover and random mutants | Yes |
| `BRKGA_CS` | `BRKGA_CS` | Biased crossover with community-based refinement | Yes |
| `GeneticAlgorithm` | `GA` | Tournament selection, crossover and mutation | Yes |
| `PSO` | `PSO` | Personal/global best velocity updates | Yes |
| `GRASP` | `GRASP` | Randomized construction, line search and RVND | Yes |
| `IPR` | `IPR` | Relink blocks between elite chromosomes | No |
| `MultiStart` | `MultiStart` | Repeated independent random candidates | No |

Algorithm selection is fixed: do not pass `algorithms=` to these constructors or
assign another selection afterward. Use `RKO` when choosing/changing a portfolio.
IPR needs a pool that can retain at least two distinct chromosomes to relink;
seeding/prefill is not feasibility repair. There is no universally best algorithm.

## Shared constructor settings

All options below have **the same names, defaults and meanings as RKO**:

| Argument | Meaning |
| --- | --- |
| `decoder`, `cost`, `n_variables` | Required problem callbacks and positive chromosome dimension (or one problem object such as `rko_solver.Callbacks`, as for `RKO`) |
| `adaptation=None` | Configured default; explicit `"fixed"` or `"adaptive"` |
| `control=None` | Existing integer control override; must agree with adaptation |
| `pool_size=None` | Configured elite capacity, not population size |
| `prefill_pool=None` | Configured shared-startup policy; currently off by default |
| `params_dir=None` | Packaged defaults or a custom TOML directory |
| `engine="python"` | `"python"`, `"rust"`, `"cpp"` or `"auto"`, exactly as for `RKO` |
| `execution=None` | Python engine only: process (default), thread or sequential |
| `n_jobs=None` | Process or Rust worker cap |
| `random_state=None` | Nonnegative library seed or fresh assigned seeds |
| `verbose=False` | Opt-in lifecycle/shared-pool output |
| `hyperparameters=None` | Nested per-algorithm scalar overrides, exactly as in RKO |
| `history_limit=0` | Disabled history, or 2–10000 retained observations per search |

Call `.solve(time_limit=...)` with positive finite seconds; it returns `RKOResult`,
not the historical kernel tuple. Defaults match RKO: **process execution** and the
configured adaptation mode, not implicit sequential/fixed settings. Use explicit
`execution="sequential", adaptation="fixed"` for simple first experiments. Process mode
requires importable callbacks and an entry-point guard; see [execution modes](usage.md#execution-modes).

## Algorithm hyperparameters are explicit

For a standalone algorithm use the **same nested dictionary** as RKO:

```python
from rko_solver.metaheuristics import SA

solver = SA(
    decoder=decode, cost=objective, n_variables=20,
    adaptation="fixed", execution="sequential",
    hyperparameters={
        "SA": {"sa_max": 100, "t0": 1000.0, "alpha": 0.98,
               "beta_min": 0.05, "beta_max": 0.20},
    },
)
result = solver.solve(time_limit=10.0)
```

There are no hidden algorithm-specific constructor keywords. The dictionary keys
are documented in the [full parameter reference](parameters.md), including defaults,
meanings and coupled constraints. Only supplied values replace packaged fixed
scalars. Unknown/unselected names and invalid values are rejected. Manual scalar
overrides require fixed adaptation; adaptive named grids remain TOML-configured.
IPR and MultiStart have no tunable search scalars and report effective control zero.

Assess either solver's result through the same optional
[`assess_result`](solution-validation.md), and generate reports through the same
[`OptimizationReport`](reporting.md). Neither loader nor validator runs inside search.

## Advanced kernel integration

The public package imports above are managed solver objects. The registry still
uses its low-level kernel classes in individual implementation modules. Those
constructors/tuple methods remain unchanged for custom-operator integration and
fixed-work diagnostics, but are **not** the normal user-facing solver interface.
For that advanced contract, import the kernel explicitly:

```python
from rko_solver.components import SolutionPool
from rko_solver.metaheuristics.sa import SA as SAKernel

kernel = SAKernel(
    SolutionPool(), decode, objective, tam_solution=10,
    sa_max=100, t0=1000.0, alpha=0.98,
)
extra, keys, cost = kernel.solve(limit_time=2.0)
```

Direct kernel execution does not activate the managed callback session, owned
seed context or immutable evaluation reports. Its defaults can differ from packaged
TOMLs. Prefer `from rko_solver.metaheuristics import SA` and `solver.solve(...)` for normal
usage. Keep custom subclasses/operators in the explicitly advanced integration
layer described in [extension guidance](extending.md).
