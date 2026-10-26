# Parameter reference

[Documentation index](index.md) · [Standalone algorithms](standalone.md)

## Fixed versus adaptive

| Public option | Legacy spelling | What actually happens |
| --- | --- | --- |
| `adaptation="fixed"` | `control=0`, called **offline** | Choose parameters before searching; values remain fixed |
| `adaptation="adaptive"` | `control=1`, called **online** | Q-learning selects feasible parameter combinations during searching |

Neither mode requires internet access. Fixed mode does not run a separate tuning
experiment, and adaptive mode is not a learned decoder. Both optimize the same
objective. Adaptive control has overhead and is **not** guaranteed to improve
quality. IPR and MultiStart have no adaptive parameter axes.

RKO and every public algorithm solver retain the same configured default with
`adaptation=None` (the packaged config currently requests adaptive control).
Choose `adaptation="fixed"` explicitly when supplying manual scalar overrides. Existing `control`
calls continue to work without warnings. The legacy name is retained for
compatibility, not removed in this change. If both names are supplied, they must
agree or a `ValueError` is raised. Validation occurs when `RKO.solve()` starts.

In TOML, `[run] adaptation = "fixed"` can replace `[run] control = 0`. Python
settings override TOML defaults. Contradictory names **within** a TOML file are
rejected. Existing `[offline]` and `[online]` table names stay unchanged.

## Set fixed values in Python

```python
from rko_solver.metaheuristics import SA

solver = SA(
    decode, objective, 20, adaptation="fixed", execution="sequential",
    hyperparameters={"SA": {"sa_max": 100, "t0": 1000.0, "alpha": 0.98}},
)
result = solver.solve(time_limit=60.0)
```

For a portfolio, nest the same settings by algorithm:

```python
from rko_solver import RKO

solver = RKO(
    decode, objective, 20, algorithms=["SA", "PSO"], adaptation="fixed",
    hyperparameters={"SA": {"t0": 1000.0}, "PSO": {"w": 0.6}},
    execution="sequential",
)
```

Only supplied values change. Installed TOMLs are never rewritten. Unknown keys,
booleans, NaN/infinity, fractional integer settings and invalid joint domains
are rejected. Scalar overrides cannot be combined with adaptive mode: configure
adaptive choices in `[online]` named TOML axes through `params_dir` instead.

## Algorithm parameters

The defaults below are the **packaged fixed TOML defaults** used by the public individual solvers
and RKO, not necessarily the defaults of direct class constructors. They are
starting points, not recommendations for every problem. Costs are assumed to be
minimized; temperature scales depend on your objective's magnitude.

### SA — simulated annealing

| Key | Default | Valid domain | Meaning and trade-off |
| --- | --- | --- | --- |
| `sa_max` | 1000 | integer >= 1 | Proposals per temperature level; more proposals spend longer at each temperature |
| `t0` | 1000000.0 | > 0 | Initial temperature scale; higher values accept more deteriorations for the same cost difference |
| `alpha` | 0.99 | 0 < value < 1 | Cooling multiplier; nearer one cools more slowly |
| `beta_min` | 0.01 | [0, 1] | Lower shaking-intensity factor relative to chromosome length |
| `beta_max` | 0.05 | [0, 1], >= `beta_min` | Upper shaking-intensity factor; larger values apply more operations |

Adaptive choices are made per temperature level. A selected `t0` governs the next
schedule, not a reset of an active temperature. See [SA control](sa-control.md).

### ILS — iterated local search

| Key | Default | Valid domain | Meaning |
| --- | --- | --- | --- |
| `beta_min` | 0.10 | [0, 1] | Minimum shaking-intensity factor before RVND refinement |
| `beta_max` | 0.20 | [0, 1], >= `beta_min` | Maximum shaking-intensity factor; larger intervals encourage escaping the incumbent basin |

### VNS — variable neighborhood search

| Key | Default | Valid domain | Meaning |
| --- | --- | --- | --- |
| `k_max` | 5 | integer >= 1 | Maximum neighborhood index before the outer loop restarts |
| `beta_min` | 0.05 | [0, 1] | Base shaking intensity; level k uses [k * beta_min, (k + 1) * beta_min] |

Shaking performs `int(n_variables * sampled_intensity) + 1` random perturbation
operations (at least one), which may touch the same coordinate repeatedly. These
are **not** exact counts of distinct changed keys. VNS scales the interval by its
neighborhood index; it can exceed one and is not capped to one by the operator.

### LNS — large neighborhood search

| Key | Default | Valid domain | Meaning |
| --- | --- | --- | --- |
| `beta_min` | 0.10 | [0, 1] | Minimum fraction destroyed before repair |
| `beta_max` | 0.30 | [0, 1], >= `beta_min` | Maximum destroy fraction; larger fractions increase repair work |
| `t0` | 10000.0 | > 0 | Metropolis temperature scale for accepting worse repaired candidates |
| `alpha_lns` | 0.95 | 0 < value < 1 | Geometric cooling factor; nearer one cools more slowly |

### BRKGA and BRKGA-CS — biased random-key genetic searches

| Key | Default | Valid domain | Meaning |
| --- | --- | --- | --- |
| `p` | 50 | integer >= 2 | Population size; larger populations cost more evaluations per generation |
| `pe` | 0.15 | [0, 1], subject to joint check | Fraction retained as elite parents (at least one elite) |
| `pm` | 0.10 | [0, 1], subject to joint check | BRKGA: random-mutant fraction; **BRKGA-CS: per-gene mutation probability** |
| `rhoe` | 0.70 | [0, 1] | Probability of inheriting an elite-parent gene during crossover |

BRKGA requires `max(1, int(p * pe)) + int(p * pm) < p` so offspring slots remain.
BRKGA-CS requires `max(1, int(p * pe)) < p`; its `pm` is not a mutant count.
Do not transfer parameter interpretations blindly between these two variants.
The direct BRKGA-CS constructor also exposes `sigma=0.6`, a positive correlation
threshold for community detection. It is not a configurable scalar in the
high-level TOML/override schema; inspect the constructor for that advanced use.

### GA — genetic algorithm

| Key | Default | Valid domain | Meaning |
| --- | --- | --- | --- |
| `size_pop` | 50 | integer >= 3 | Population size for three-way tournament selection |
| `prob_cros` | 0.80 | [0, 1] | Pair-level crossover gate, not elite inheritance probability |
| `prob_mut` | 0.02 | [0, 1] | Probability of replacing each mutated gene with a new random key |

### PSO — particle swarm optimization

| Key | Default | Valid domain | Meaning |
| --- | --- | --- | --- |
| `p_size` | 30 | integer >= 1 | Number of particles, not elite-pool size |
| `c1` | 2.0 | >= 0 | Personal-best attraction coefficient |
| `c2` | 2.0 | >= 0 | Global-best attraction coefficient |
| `w` | 0.70 | >= 0 | Inertia/constriction multiplier in this implementation's velocity formula |

The implemented update is `V = w * (V + c1*r1*(Pbest-X) + c2*r2*(Gbest-X))`,
not a guarantee of the common textbook `w*V + ...` variant. Out-of-range gene
positions revert to their previous values and their velocity components become zero.

### GRASP — construction plus local refinement

| Key | Default | Valid domain | Meaning |
| --- | --- | --- | --- |
| `hs` | 0.125 | > 0 | Initial line-search step size; coarser steps reduce grid resolution |
| `he` | 0.00098 | > 0 and <= `hs` | Final step threshold; smaller values allow finer search and more work |

The adaptive `alpha_grasp` axis is retained for reference state-graph compatibility
but does **not** control a restricted candidate list or directly change construction.
It is not a fixed scalar override. See [alpha contract](grasp-alpha.md) and
[line-search behavior](grasp-line-search.md). The step is halved after construction
fails to improve the global best, not unconditionally on every iteration.

### IPR and MultiStart

Neither exposes tunable fixed/adaptive search hyperparameters. IPR depends on
elite diversity and enough pool capacity to retain a pair; MultiStart simply
samples independent random keys. Neither can guarantee feasibility or optimality.

## Population, pool and budget are different

`pool_size` controls shared elite retention; `p`, `size_pop` and `p_size` control
algorithm populations. They are not interchangeable. A larger population can
consume a short budget before completing a generation. Likewise, adaptive grids
with many combinations can add initialization/control overhead. Benchmark choices
with your actual decoder rather than assuming larger means better.

Inspect `result.searches[0].parameters` and `.parameter_grid` for validated initial
settings and adaptive axes. They are snapshots, not final parameters or a history
of Q-learning state transitions. See [result accounting](usage.md#result-accounting-and-configuration-snapshots).
