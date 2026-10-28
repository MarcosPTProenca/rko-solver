# Optional static decoder diagrams

[Documentation index](index.md) · [Reports](reporting.md) · [Problem pipeline](problem-adapters.md)

Random keys can make ordering and construction rules visible **when your decoder
expresses understandable rules**. This feature helps communicate those rules. It
does not make every decoder interpretable or prove that RKO is more interpretable
than another optimizer.

## Three different kinds of explanation

| Category | Available here | Meaning and limit |
| --- | --- | --- |
| Author-declared logic | Yes | The rules you state; correspondence with code is not verified. |
| Source-derived structure | Yes, supported Python subset | Syntax/branches/loops with opaque calls; not complete business semantics. |
| Observed decisions | No | Requires separate instrumentation during the actual evaluation. |

Neither static option is a trace of the winning evaluation. Re-decoding keys later,
including [post-run assessment](solution-validation.md), remains reconstruction,
not the original witness. Diagrams do not certify feasibility, optimality,
constraint coverage or the behavior of called libraries.

## Python first: explicitly inspect a decoder

Normal solving needs no diagram, decorator, adapter inheritance or source access.
After an ordinary solve, explicitly request a graph and pass it to the report:

```python
from pathlib import Path
from rko_solver import RKO
from examples import instance_pipeline
from rko_solver.explain import DecoderDiagram
from rko_solver.reporting import OptimizationReport

path = Path(instance_pipeline.__file__).with_name("instances") / "scheduling.json"
instance = instance_pipeline.read_data(path)
problem = instance_pipeline.SchedulingProblem(instance)
solver = RKO(
    problem.decode, problem.cost, problem.n_variables,
    algorithms=["SA"], adaptation="fixed", execution="sequential", random_state=42,
)
result = solver.solve(time_limit=0.1)
diagram = DecoderDiagram.from_callable(problem.decode)
report = OptimizationReport(result, decoder_diagram=diagram)
report.write_json("decoder.json")
```

To write HTML, install the optional reports extra and call
`report.write_html("decoder.html")`. The report reuses its **existing embedded
Plotly bundle**, with no Mermaid/Graphviz installation, browser automation, server,
CDN or source-upload service. It renders a bounded directed graph with arrow/branch
labels and full readable step/transition tables. Layout uses simple depth bands,
not an optimized general graph-layout engine; complex graphs may need zooming or
the text alternative. The numerical analysis remains separate and unchanged.

For a complete HTML/JSON example including optional independent checks:

```bash
uv sync --locked --extra reports
uv run --locked --extra reports python -m examples.decoder_diagram
```

It writes ignored local `results/decoder-diagram.html` and `.json`, not benchmark
evidence. Source inspection and rendering happen **after solving**, outside
`result.elapsed_seconds` and evaluation counts. The diagram builder never invokes
the decoder/objective. `OptimizationReport` never inspects source automatically.
Optional assessment in the example separately decodes/checks once, as documented.

## Supported source and explicit limitations

`DecoderDiagram.from_source(source, function_name="decode")` accepts explicitly
provided source without executing module statements, function bodies, decorators,
default expressions or annotations. Select one top-level synchronous function; if
only one exists, its name may be omitted. `from_callable` obtains source through
`inspect.getsource`, handling its documented TypeError/OSError; partials, Python
bound methods and inspectable Python callable objects are supported. For Python
functions, the helper inspects the actual code object, not copied `__wrapped__`
metadata that could substitute a decorated function's original body. Decorator
implementation/transitive calls are not expanded.

Supported body statements are assignments/updates/annotations, opaque expressions
and calls, `if`/`elif`/`else`, `for`/`while` without loop-`else`, `break`, `continue`,
`return`, explicit `raise`, and `pass`. Loop jumps target the nearest enclosing
loop. Unconditional return/raise/jump has no invented fallthrough edge. Explicit
raises are error terminals, not successful returns.

**Expressions and calls remain opaque**: no expansion of short-circuit operators,
comprehension loops, library internals, dynamic dispatch or implicit exceptions.
Both syntactic branches are represented without proving predicate feasibility.
The diagram omits trailing statements after unconditional termination; it does
not perform general dead-code or reachability analysis. A source snapshot/hash
is not proof of what code executed during an earlier optimization.

Unsupported control (including try/with/match, nested definitions, yield-based generator functions,
async/await and loop-else) fails with a `ValueError` naming the construct/line;
it is not silently replaced by a misleading successful path. Built-ins, native/JIT
code, frozen artifacts and unavailable notebook source can fail inspection. Provide
explicit source when available or a declared graph instead. No transitive call
analysis, runtime instrumentation, automatic callback replay or LLM interpretation
is performed.

## Author a domain-meaningful graph instead

For a capacity-limited selection decoder, a declaration can communicate rules that
source syntax alone cannot. This example is **not** a declaration about the separate
scheduling example above; attach it only to a result whose decoder follows these rules:

```python
from rko_solver.explain import DecoderDiagram, DiagramEdge, DiagramNode

capacity_diagram = DecoderDiagram(
    title="Priority selection with a capacity check",
    description="Author-declared construction rules; not verified against code.",
    nodes=[
        DiagramNode("keys", "Random keys", "start"),
        DiagramNode("rank", "Order candidates by key"),
        DiagramNode("candidate", "Another candidate?", "decision"),
        DiagramNode("capacity", "Fits remaining capacity?", "decision"),
        DiagramNode("accept", "Accept and update capacity"),
        DiagramNode("skip", "Skip candidate"),
        DiagramNode("solution", "Build selected solution"),
        DiagramNode("end", "Return solution; optional checks occur afterward", "end"),
    ],
    edges=[
        DiagramEdge("keys", "rank"), DiagramEdge("rank", "candidate"),
        DiagramEdge("candidate", "capacity", "yes"),
        DiagramEdge("candidate", "solution", "no"),
        DiagramEdge("capacity", "accept", "yes"),
        DiagramEdge("capacity", "skip", "no"),
        DiagramEdge("accept", "candidate", "next"),
        DiagramEdge("skip", "candidate", "next"),
        DiagramEdge("solution", "end"),
    ],
)
```

Attach through `OptimizationReport(result, decoder_diagram=capacity_diagram)`
with the **matching** result. This association is application-owned, not verified
by the library. Existing solvers and decoder signatures do not change. Diagrams
can be omitted entirely. Declared/source-derived metadata cannot claim an
`observed` provenance category.

For events emitted during an actual evaluated decoder invocation, use the separate
[observed-decisions API](decision-traces.md). Static graph provenance is unchanged;
a graph must not be relabelled as an observed execution.

## Bounds, data ownership and privacy

- Graphs own immutable node/edge tuples: **100 nodes**, **200 distinct edges**,
  240-character node labels, 80-character edge labels, 160-character title,
  2000-character description. IDs are bounded ASCII identifiers. Exactly one start,
  existing edge endpoints, reachable nodes and at least one end/error are required.
  Start has no incoming edges; end/error terminals have no outgoing edges.
  Nonterminals need outgoing transitions. Cycles are allowed. These structural checks are not domain validation.
- Explicit source is limited to **64000 characters** and **4000 AST nodes**;
  conversion also respects graph limits. Excessive/invalid source fails before
  rendering. Generated labels can be abbreviated.
- Labels are plain text, escaped in HTML/Plotly. No executable HTML/link/script,
  user SVG, external converter or automatic source upload is accepted.
- Source-derived JSON records origin, selected function name, relative snippet
  lines, SHA-256 and expression-disclosure flag. **No source file paths, complete
  raw-source field, bound instance values or callback arguments** are exported.
  This excludes automatic file-path metadata, not paths you put in authored labels
  or explicitly disclosed source literals.
- By default, labels name identifiers/callees but omit source expressions/literals.
  Identifiers and authored labels can still reveal business IP: review before sharing.
- `include_expressions=True` explicitly opts into abbreviated source expressions,
  **including literals**. This can disclose secrets; it is never enabled automatically.
  Do not put credentials in decoder source or opt in without reviewing outputs.
- Report JSON schema 1 adds nullable `decoder_diagram`; absent means no diagram,
  not missing/failed validation. Diagram absence causes no source inspection.

Static source inspection operates on trusted application Python objects, not a
sandbox for executing arbitrary objects. No original-evaluation instrumentation,
formal verification or comparative interpretability evidence is included.
