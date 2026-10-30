# RKO documentation

RKO minimizes an objective over random-key vectors. You provide the instance data,
a decoder and an objective; the library provides the searches and cooperating
portfolio. Domain feasibility and final validation remain your responsibility.

This is the **development site for `main`**, not a versioned manual for every
published package. Start with [installation and version boundaries](installation.md).
The [current limitations](status.md) distinguish implemented behavior, alpha-release
boundaries and scientific claims that have not been established.

## Start here

1. [Choose and install a version](installation.md): source versus a published package,
   optional extras, platform artifacts and free-threading limits.
2. [Run a first solve](getting-started.md): one algorithm, explicit execution and result decoding.
3. [Load your own instance](problem-adapters.md): bind application data to ordinary callbacks
   and optionally check the final solution. No adapter base class is required.
4. [Configure the solver](usage.md): engine selection, execution, budgets, results and errors.

## Choose a workflow

| Goal | Maintained guide |
| --- | --- |
| Select an individual algorithm | [Standalone solvers](standalone.md) |
| Set fixed scalars or adaptive grids | [Parameter reference](parameters.md) |
| Run compiled, block or built-in objectives | [Rust and C++ engines](native.md) |
| Run from the terminal | [Repository example runner](running.md) |
| Understand workers, startup and resource limits | [Process workers](process-workers.md), [initialization](driver-initialization.md), [native threads](native-threads.md) |
| Convert a bounded mathematical model | [Model contracts](modeling.md), [historical qualification](model-validation.md) |
| Check a final solution | [Optional validation](solution-validation.md) |
| Export offline analysis | [Reports](reporting.md) |
| Illustrate declared decoder structure | [Static diagrams](decoder-diagrams.md) |
| Retain authored decisions from original evaluations | [Observed decisions](decision-traces.md) |
| Bind trusted callbacks to a private HTTP service | [Service and scaffold](service.md) |
| Explore bounded decoder generation and held-out studies | [Experimental evolution](self-evolve.md), [locked studies](confirmation-study.md) |
| Find the papers behind an algorithm, a component or an experiment | [Scientific references](references.md) |

## Understand the limits

[Execution accounting](usage.md#execution-modes), [convergence guarantees](guarantees.md),
[C++ compatibility](cpp-compatibility.md) and [performance criteria](performance.md)
answer different questions. Passing installation tests does not establish solution
quality, universal thread safety, timing guarantees or general C++ parity.
Identified measurements and their original provenance stay in the
[benchmark reference](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md).

## Extend and maintain

- [Architecture](architecture.md): public contracts, source layout and enforced import boundaries.
- [Extension guide](extending.md): replace operators or implement an algorithm.
- [Contributing](https://github.com/MarcosPTProenca/rko-solver/blob/main/CONTRIBUTING.md): tests,
  quality checks, evidence rules and review expectations.
- [Documentation site](documentation-site.md): local preview, publication boundaries and recovery.
- [Controlled releases](https://github.com/MarcosPTProenca/rko-solver/blob/main/RELEASING.md):
  owner-approved publication, not an automatic consequence of merging code.
