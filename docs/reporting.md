# Interactive HTML analysis reports

[Documentation index](index.md) · [Standalone algorithms](standalone.md)

Generate a **self-contained interactive HTML file** after optimization. It opens
from disk without a server, internet access, remote fonts or a CDN. Hover over
observations, zoom, and click legend entries to hide/show search curves.

## Install the optional extra

From the source checkout:

```bash
uv sync --locked --extra reports
```

Plotly is optional: ordinary optimization still requires only NumPy. Importing
`rko` or exporting report JSON does not import Plotly. HTML rendering requires the
extra; it does not install or start Chrome. This development API is not a new PyPI
release; use the checked-out repository and its lockfile.

## First report

Using the decoder/objective from [getting started](getting-started.md):

```python
from rko_solver.metaheuristics import SA
from rko_solver.reporting import OptimizationReport

solver = SA(
    decode, objective, n_variables=10,
    adaptation="fixed", execution="sequential", random_state=42, history_limit=1000,
)
result = solver.solve(time_limit=2.0)
report = OptimizationReport(result, title="My optimization run")
report.write_html("run.html")
report.write_json("run.json")
```

`write_html` and `write_json` return the written `Path`. The parent directory must
exist; existing targets are atomically replaced only after rendering succeeds.
Export failures raise without modifying the successful optimization result.

For a complete runnable example:

```bash
uv run --locked --extra reports python -m rko_solver.examples.analysis_report
```

It writes `results/analysis-report.html` and `.json`. These are generated local
artifacts, not maintained benchmark evidence. Rendering happens **after** the
solve, outside `result.elapsed_seconds`. Large embedded Plotly bundles make HTML
files several megabytes even for small runs; the library is embedded only once.

## Portfolios use the same API

```python
from rko_solver import RKO

result = RKO(
    decode, objective, 10, algorithms=["SA", "PSO"],
    adaptation="fixed", execution="sequential", random_state=42,
    history_limit=1000,
).solve(2.0)
OptimizationReport(result).write_html("portfolio.html")
```

Process/thread modes support the same history setting. Process mode still needs
importable callbacks and a guarded entry point. Sequential mode gives each algorithm
its own budget; process assignments share a deadline. These are not interchangeable
wall-time experiments. See [execution modes](usage.md#execution-modes).

## What is included

- Run summary: minimized cost, evaluations, end-to-end elapsed time, engine and
  execution, requested mode, seeds, dimensions, worker settings and stop reason.
- Interactive best-so-far curves against **local search seconds** and local
  objective-evaluation index, when histories were recorded.
- Per-search evaluation counts and assigned versus measured search duration.
- Initial scalar parameters and adaptive-grid axes; not a Q-learning trajectory.
- Separate shared-initialization accounting and runtime-version provenance.
- Optional, explicitly user-supplied domain values such as profit or feasibility.

Plot data is also represented by an accessible HTML table for summary/accounting;
individual history samples remain in the versioned embedded JSON. The JSON export
contains the same data without requiring a browser or Plotly. Schema version 1
contains `summary`, `searches`, `initialization`, `versions` and `domain_metrics`.
An additive `assessment` section records optional [post-solve checks](solution-validation.md),
explicit feasibility and separate postprocessing time; without a checker it says
`not_checked`. `OptimizationReport(result, assessment=assessment)` exports only
diagnostics, not the decoded solution or source keys. User metrics are not promoted
to validation results. It is a data-only artifact, not a replay script. This version does not provide
JSON import, automatic experiment aggregation or programmatic PDF conversion.

## Optional static decoder logic

Explicitly request a diagram after solving and attach it separately:

```python
from rko_solver.explain import DecoderDiagram

diagram = DecoderDiagram.from_callable(decode)
report = OptimizationReport(result, decoder_diagram=diagram)
report.write_html("decoder.html")
```

An author-declared `DecoderDiagram(nodes=..., edges=...)` is an alternative when
source is missing or domain meaning is more useful than syntax. Both render with
the already embedded Plotly library and complete plain step/transition tables.
The report labels origin and disclosure flags, never inspects code automatically,
and never invokes callbacks to draw the graph. JSON schema 1 gains nullable
`decoder_diagram`, containing graph metadata rather than complete raw source or
callback data. No extra renderer dependency or remote service is introduced.

Static graphs are **not execution traces**, proofs of decoder completeness, or
explanations of the original winning evaluation. The AST subset excludes unsupported
control and keeps expressions/calls opaque. Default labels omit literals, but code
identifiers and user labels may still be sensitive. Expression disclosure is
explicitly opt-in. Read [supported syntax, limits, provenance and privacy](decoder-diagrams.md)
before inspecting or sharing a decoder. For separately opted-in authored events from an original evaluation, see
[observed decoder decisions](decision-traces.md). These are not static diagrams;
recorded, truncated and unavailable evidence remain visibly distinct.

## Bounded, opt-in history

`history_limit=0` (default) disables capture. A positive integer from **2 to 10000**
is a per-search cap, not a cap on all searches combined. A suggested first value
is 1000. Each observation contains only time, completed evaluation count and the
best objective; it stores no chromosomes, decoded payloads or populations.

The recorder samples incumbent improvements and the terminal session observation.
If the cap is exceeded, it deterministically drops alternating interior samples,
retaining first and latest observations. `history_observations` counts all observed
samples including terminal observations; it is **not** an improvement count.
`len(search.history)` is the retained count. When observations exceed retained
samples, the curve is labelled **compacted** and its intermediate step positions
are approximate. No exact target-crossing times are inferred from that curve.

Retained work is bounded by the number of searches times the selected limit;
worker histories transfer at completion, not through a manager call per objective.
Collecting timestamps and samples still adds overhead and can affect timed results.
Leave capture disabled for strict performance studies unless measuring that overhead.
No RAM/CPU sampling, every-evaluation storage or telemetry threads are started.

Without history, reports still show summary/configuration/work charts and explicitly
mark convergence as unavailable. They never invent a curve from the final result
or reconstruct observations by running callbacks again.

## Interpret the plots correctly

- Each curve uses its **own** search-start clock. The curves are not a synchronized
  portfolio timeline or the shared-pool publication history.
- Shared prefill and worker queueing are excluded from search curves. Shared
  initialization is reported separately and can produce the winning final result.
- A terminal incumbent can be observed after its nominal allowance because stopping
  is cooperative. Its actual timestamp is retained, not moved inside the budget.
- Concurrent session durations overlap; summing them does not give wall time.
- Evaluations/second is not CPU utilization or equivalent work across objectives.
- Portfolio searches exchange elites. Curves and counts do not establish independent
  algorithm rankings, causal contribution, statistical significance or convergence.
- `completed` means execution returned, not a global-optimality certificate.

To compare algorithm quality reliably, run independent repeated experiments on
representative instances with explicit budget policies. Single-run reports contain
no boxplots, confidence intervals or significance tests pretending to do that work.

## Domain metrics and privacy

Pass values you have already computed and validated yourself:

```python
report = OptimizationReport(
    result,
    title="Production scheduling",
    metrics={"Feasible": True, "Total tardiness": 14.0, "Time unit": "minutes"},
)
```

These are examples of **caller-supplied labels**, not values inferred by the solver.
Finite penalized objectives do not prove feasibility, and minimized cost is not
necessarily negative profit once penalties are included. No percentage gap is
invented for absent, zero or negative reference values. The report never calls
your decoder/objective to infer domain metrics.

Titles/labels are escaped, embedded JSON protects script-closing sequences, and
only string labels with finite scalar values are accepted for domain metrics.
Callbacks, keys and input data are excluded by default, but objectives, seeds,
parameter values and your supplied metrics are included. Review these before
sharing the file; self-contained does not mean free of confidential information.
Do not put secrets in titles or metrics.

## Current boundaries

- HTML and JSON only. Automatic PDF export would require a separately designed
  renderer and dependency policy; this version does not download Chrome.
- No live dashboard, remote service, stored populations, memory sampler or adaptive
  state-history heatmap.
- No recording of failed/aborted optimization runs: existing exception behavior remains unchanged.
  Failed domain checks on completed runs can be included through an optional assessment.
- Low-level direct constructor `.solve()` calls do not activate managed history.
  Use public algorithm solver objects or RKO, followed by `.solve(...)`, for reportable metadata.
