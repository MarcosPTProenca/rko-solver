# Current limitations

Reviewed on **2026-10-29** for development `main`. This is an alpha library, not a
stable API or a universal optimization-performance guarantee. The published
[0.3.0a4 prerelease](https://pypi.org/project/rko-solver/0.3.0a4/) predates some current
source changes; a matching package-version string does not make their contents
identical. See [installation and version boundaries](installation.md).

## Implemented and functionally checked

- **Public solver lifecycle:** `RKO` and the eleven individual solver classes configure
  without starting a search, return immutable results and build fresh state per solve.
  Canonical `engine` and `execution` settings are independent. Development `main` removes
  deprecated `backend` arguments/properties, alternative constructors and import shims;
  see [maintained replacements](architecture.md#canonical-public-api).
- **Three engines:** Python, Rust and the optional C++ reference engine support the
  documented problem styles and result accounting. The default remains the Python
  engine with process execution. Built-in and Numba problems need a compiled engine;
  ordinary callbacks and block objectives have their own concurrency limits.
- **Configuration and stopping:** named parameters, joint validation, library-owned
  random streams, completed-objective counts, per-search metadata and best-incumbent
  retention have regression coverage. Caller-provided `initial_keys` are evaluated before
  search and reported separately from random prefill; see [seed accounting](usage.md).
  [Protected exploration](guarantees.md) has a
  specific mathematical guarantee; it is not a rate or solution-quality guarantee.
- **Installation:** regular artifacts and source installations are checked on the
  declared platform matrix. Linux free-threaded CPython 3.14 has a separate source-install
  smoke that rejects interpreter-lock fallback. This does not qualify every optional
  dependency, future Python release or free-threaded artifact on other platforms.
- **Application integration:** file-backed callback examples, optional final-solution
  validation, reports, static diagrams and authored decision traces have executable
  examples and focused tests. Repository examples now live under `examples/`, outside
  current development distributions; the old installed demo command is removed.
- **Bounded mathematical models:** `RKO.from_model` has functional and installed-extra
  checks for pinned Pyomo, PuLP and MathOpt frontends. The current continuous-restoration,
  source-identity and independent-witness contracts are documented in [modeling](modeling.md).
  Historical performance records remain tied to their original source, not automatically
  requalified by the current hosted tests or renamed package.

The maintained [continuous-integration workflow](https://github.com/MarcosPTProenca/rko-solver/blob/main/.github/workflows/ci.yml)
and [artifact workflow](https://github.com/MarcosPTProenca/rko-solver/blob/main/.github/workflows/wheels.yml)
define executable checks. Their latest results must be read for the exact source
revision being reviewed; this page is not an assertion that every future run passes.

## Core limits

| Area | Boundary |
| --- | --- |
| Callback behavior | Callbacks must return valid finite values and manage their own randomness, mutation and concurrent access. Library-owned seeds do not seed callback-global generators. |
| Solver concurrency | Independent instances may overlap. Do not call `solve` concurrently on the same mutable instance. |
| Time limits | Deadlines are cooperative. An active callback can overrun; blocked Python threads cannot be safely killed. Process watchdogs do not make this a hard real-time API. |
| Reproducibility | Fixed seeds do not remove timed scheduling, elite-exchange or callback-state differences. Evaluation limits bound work but do not make all concurrent configurations deterministic. |
| Worker resources | Process discovery and `n_jobs` do not cap numerical-library threads, independent solvers or every container quota. Oversubscription remains possible. |
| Provenance | Result snapshots do not capture every callback/data/source identity or adaptive trajectory. They are not a complete replay artifact. |
| Scientific quality | Matching names, nominal budgets or pool capacities does not prove equal work, equal quality or universal C++ parity. |

[Worker accounting](process-workers.md), [native-thread limits](native-threads.md),
[callback/stopping contracts](usage.md#adapting-a-custom-problem) and
[C++ differences](cpp-compatibility.md) describe those boundaries in detail.
Algorithm implementations intentionally differ in some initialization, random
streams, adaptation and neighborhood policies; no unchanged-trajectory promise is made.

## Optional-feature limits

- **Model conversion is bounded, not a complete mathematical-programming solver.**
  It supports the documented linear frontend subset, requires finite effective bounds
  and uses only the Python engine. Continuous restoration does not solve arbitrary
  integer feasibility. Independent feasible witnesses do not certify optimality.
  Fresh source-specific scientific qualification remains separate from functional tests;
  see [historical model measurements](model-validation.md).
- **Solution assessment checks a post-run reconstruction.** It does not establish that
  a stochastic or stateful decoder reproduced the winning evaluation, or that a numeric
  objective encodes all domain constraints. See [validation](solution-validation.md).
- **Explanations are bounded.** Static diagrams are declarations or best-effort source
  structures, not execution traces. Recorded decisions are partial author-emitted events
  retained only when original evaluation identity is intact. Neither proves domain
  completeness, feasibility or optimality. See [diagrams](decoder-diagrams.md) and
  [observed decisions](decision-traces.md).
- **The private service is a Linux-container reference.** It binds trusted installed
  callbacks, one authenticated job at a time, with ephemeral bounded output and
  supervised processes. Runtime resource limits are still required. It is not a public
  multi-tenant service, durable queue or hostile-code sandbox; native Windows supervision
  is unsupported. See [service requirements](service.md).
- **Decoder evolution is experimental.** The pilot uses review-gated proposals,
  isolated evaluation and original-source witness checks. Fake-process tests do not
  qualify live provider access, terms, account eligibility or isolation. Broader runtime
  qualification, sealed-test confirmation and production promotion remain separate.
  See [the pilot](self-evolve.md) and [locked-study tooling](confirmation-study.md).

## Measurement boundaries

Performance evidence belongs to its identified source, data, parameters, hardware,
seeds and budgets. The [benchmark reference](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md)
preserves raw measurements and reports losses as well as wins. The observed native
engine results cover declared Knapsack, TSP and Rastrigin regimes, not arbitrary
callbacks or application families. Scalar callbacks, block objectives and compiled
objectives have materially different costs.

Kernel throughput, traced allocations, sampled process memory, startup latency
and quality-at-time are different measurements. None substitutes for the others.
The [expanded prefill study](prefill-expanded-work.md) includes regressions and
startup-only wins, so prefill remains opt-in. Historical Knapsack comparisons
predate later corrections; do not merge their rows with new-source campaigns.

Use [performance criteria](performance.md) for new claims. Do not infer a general
speedup from a language choice, passing tests, a synthetic example or one machine.
No new performance claim follows from reorganizing source or documentation.

## Releases and maintenance

Published prereleases are immutable, owner-approved snapshots. A source merge or
a successful documentation deployment does not publish a package or imply stable
compatibility. Future releases need fresh versioning, artifact verification and
owner approval under the [release procedure](https://github.com/MarcosPTProenca/rko-solver/blob/main/RELEASING.md).

The required licenses and attribution remain in
[LICENSE](https://github.com/MarcosPTProenca/rko-solver/blob/main/LICENSE) and
[NOTICE](https://github.com/MarcosPTProenca/rko-solver/blob/main/NOTICE). The owner's
recorded distribution attestation is not independent legal certification; review
new third-party additions separately. Internal audits and session plans are not
public product documentation.
