# Documentation

RKO is a random-key **minimization** library. Implement a decoder and an objective
once; use them with any built-in algorithm or a cooperating portfolio. It is a
development library, not a claim of release readiness or C++ equivalence.

## Learn

Development port: [bounded model conversion](modeling.md) and
[historical qualification](model-validation.md). Current source uses `rko_solver`
and explicit Python engine/execution settings; historical measurements are not
relabeled as qualification of the renamed port.

1. [Getting started](getting-started.md): solve your first problem with one algorithm.
2. [Read instance data and bind callbacks](problem-adapters.md): a complete file-to-solution tutorial, representations and constraints.
3. [Parameter reference](parameters.md): choose fixed or adaptive settings and understand each knob.

[Controlled releases](https://github.com/MarcosPTProenca/rko-solver/blob/main/RELEASING.md)
are owner-approved, not automatic publication from main.

## Apply and extend

- [Standalone algorithms](standalone.md): all eleven classes, tuning and result metadata.
- [Optional solution validation](solution-validation.md): final domain checks and report diagnostics.
- [Interactive reports](reporting.md): offline HTML charts, bounded histories and JSON export.
- [Static decoder diagrams](decoder-diagrams.md): optional declared logic or supported Python structure, not execution traces.
- [Observed decoder decisions](decision-traces.md): bounded opt-in authored events from an original evaluation, without replay.
- [Private FastAPI/Docker service](service.md): optional authenticated, bounded Linux reference and no-overwrite scaffold.
- [Documentation website](documentation-site.md): local preview, publication boundaries and Pages setup.
- [Architecture](architecture.md): the public API, the three engines and where each lives.
- [Extension guide](extending.md): replace local search/shaking or implement a new algorithm.
- [Portfolio and execution API](usage.md): sharing elites, processes, budgets and reproducibility.
- [Rust and C++ engines](native.md): the compiled engines, problem styles and their limits.
- [Terminal usage](running.md): the supplied Knapsack adapter and custom TOML files.
- [Contributing](https://github.com/MarcosPTProenca/rko-solver/blob/main/CONTRIBUTING.md): development setup, tests and release gates.

## Investigate

- [Limitations](status.md) and [C++ compatibility](cpp-compatibility.md).
- [Measurements](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md), not general performance promises.
- Algorithm module docstrings describe technical strategy, pseudocode and stopping
  criteria. Public classes such as `SA` share RKO's constructor options and solve
  contract: read `help(SA)` and `help(SA.solve)` for managed usage. Explicit kernel
  imports are reserved for advanced operator integration.

Guides use `adaptation="fixed"` and `adaptation="adaptive"` rather than legacy
`control=0` / `control=1`. These names describe parameter selection, **not** network
connectivity, model training, execution mode, or a different objective.
