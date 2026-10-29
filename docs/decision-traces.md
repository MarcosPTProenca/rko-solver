# Optional observed decoder decisions

Record **explicit author-emitted events during the original decoder invocation**
associated with a completed finite objective. This is different from a
[declared/static decoder diagram](decoder-diagrams.md) or a
[post-run reconstruction/check](solution-validation.md).

Recording is off by default and adds no dependency beyond core NumPy. It does not
inspect source, infer business semantics, replay callbacks, capture arbitrary
payloads or certify feasibility/optimality. Uninstrumented callbacks still work.

## Run the maintained example

The existing scheduling adapter emits bounded `priority_order` and `append_job`
events at their actual execution points, without automatically exporting instance
values. Enable recording through the ordinary solver:

```bash
uv run --locked python -m examples.decision_trace
uv run --locked --extra reports python -m examples.decision_trace --html results/observed-decisions.html
```

The first command needs only core dependencies. The second adds an offline report;
report creation does not execute the decoder or objective again.

## Instrument your decoder explicitly

Keep the same three-function application contract. Emit only text you intentionally
want to expose, inside the decoder at the actual decision point:

```python
import numpy as np
from rko_solver import RKO
from rko_solver.tracing import TraceConfig
from rko_solver.tracing import record_decision


def decode(keys: np.ndarray) -> tuple[int, ...]:
    """Construct a stable priority permutation and emit an optional authored decision.

    Args:
        keys (np.ndarray): Original candidate chromosome.

    Returns:
        tuple[int, ...]: Priority order of item indices.
    """
    order = tuple(int(index) for index in np.argsort(keys, kind="stable"))
    record_decision("priority_order", "Use increasing key order with stable ties.")
    return order


def objective(order: tuple[int, ...]) -> float:
    """Evaluate the original priority payload without invoking the decoder again.

    Args:
        order (tuple[int, ...]): Original decoded permutation.

    Returns:
        float: Illustrative finite minimization objective, not domain feasibility.
    """
    return float(sum(position * item for position, item in enumerate(order)))


def main() -> None:
    """Run bounded original-decoder recording with ordinary unchanged callback signatures."""
    result = RKO(
        decode,
        objective,
        4,
        algorithms=["SA", "PSO"],
        execution="sequential",
        random_state=42,
        trace=TraceConfig(max_events=128, max_bytes=65536, max_records=8),
    ).solve(time_limit=0.2)
    print(result.trace.to_dict())


if __name__ == "__main__":
    main()
```

`record_decision(action, reason="")` is a no-op outside an active opted-in decoder.
The objective receives the exact original decoder payload; objective execution
establishes the observation's finite cost but its events are not implicitly recorded.
Ordinary standalone classes support the same `trace=TraceConfig(...)` argument.
Sequential, legacy thread and process execution are covered, including limited process
workers and the separately accounted pool initializer.

Callbacks must emit from their synchronous invocation context. This is not a
transitive interpreter/native/thread event logger; background tasks, separately
spawned callback threads and opaque external libraries are not automatically traced.

## Identity, not a matching-looking reconstruction

The engine retains an internal observation reference with evaluated costs through
its ordinary algorithm caches, incumbent selection, elite admissions and owned
process transport. A trace is attached only through the **selected reference**.
Coordinate/cost equality is never used to search for an alternative witness.

This matters when repeated keys/equal costs have different stochastic/stateful
payloads and decisions. A numerical binding checksum verifies that a reference
was not paired with changed coordinates/cost; it is **not** an execution signature
or authenticity certificate. Opaque evaluation IDs are intentionally unique and
not deterministic run-reproduction identifiers; creating them does not consume
optimizer/application RNG streams.

`result.trace` is `None` when disabled. When enabled, inspect its explicit state:

| State | Meaning |
| --- | --- |
| `recorded` | Events came from that preserved original evaluation reference |
| `no_events` | No explicit events were emitted; not proof that no decisions occurred |
| `truncated` | Count/encoded-byte limits omitted events; inspect `dropped_events` |
| `unavailable` | Identity was lost, event content evicted or numerical binding changed |

Unavailable reasons are `identity_unavailable`, `retention_limit` or
`binding_mismatch`. The engine does not alter numerical selection/tie handling just
to find a recorded trace. Custom operators that coerce a tagged internal cost to an
ordinary scalar or synthesize a new cost can discard identity: unavailable is the
honest result, even if another recorded observation has identical keys/cost.

Decoder/objective errors and nonfinite objectives retain existing exception
behavior and do not create successful witnesses. An interrupted decode without a
completed finite objective is not a result observation. Contexts are reset after
failures/nested calls so events cannot silently enter the next invocation.

## Explicit resource and privacy bounds

Defaults: at most **128 events**, **64 KiB UTF-8 JSON event-array bytes** per witness,
and **8 live event-payload records per search session**. Actions are nonblank plain
strings up to 256 characters; reasons up to 1024. The supported configurable envelope
is 1–1024 events, 64–1048576 event-array bytes and 2–64 live session records.

The session protects its current incumbent and evicts old unpinned content. An
algorithm retaining many numeric population references does not retain an event
archive for each: evicted references explicitly lose their event payload. Accepted
elite evidence is detached into the existing capacity-bounded pool; result evidence
is immutable and independent of later session activity. Process transfers carry
bounded immutable snapshots, not callbacks/recorders/payloads.

Total memory is not simply 64 KiB: account for session records, configured pool
capacity, retained results, worker count and bounded in-flight transport snapshots.
Local/process elite-reader references join the same active session record budget;
retaining many imported numeric costs does not bypass eviction. A conservative event
content bound is `(search sessions × max_records + pool capacity + retained result
snapshots + in-flight snapshots) × max_bytes`, plus bounded metadata/Python object
overhead and ordinary candidate arrays. There is no all-evaluation archive. Applications
that retain many completed results still own that additional storage.

Count/byte exhaustion truncates events rather than extending search allowances.
Malformed active labels raise `ValueError`; disabled calls do not validate/log them.
Only authored action/reason text is exported. No automatic input values, decoded
payload, source, locals, filesystem names, credentials or private timestamps are
captured. **Authors can still disclose sensitive information by putting it in text.**
Review event labels before sharing results; escaping is not privacy redaction.

## Reports and overhead

`OptimizationReport(result)` includes an additive `decision_trace` JSON field and
an ordered, escaped offline HTML event table. It labels `original_evaluation`,
identity, state and omitted-event count. It does not invent unobserved branches or
turn a partial ordered list into a complete flowchart. Declared/source diagrams and
post-run assessments remain separate sections. A manually constructed numerical
result cannot attach evidence with a different binding unnoticed by the report.

Tracing consumes real decoder/evaluation/transport time **inside the existing
budget**. It can reduce completed work and change timed search outcomes. No zero-
overhead, timed reproducibility or generic performance improvement is claimed.
For very cheap callbacks, serialization/checking overhead can dominate the objective.

Run equal-work cheap/heavier callback measurements with separate timing and Python
allocation passes:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python -m benchmarks.runtime.decision_traces --evaluations 1000 --dimension 64 --repeats 3
```

The JSON reports raw disabled/enabled times, equal completed evaluation counts and
Python `tracemalloc` peaks. These are scoped recording-overhead observations, not
search-quality measurements, total RSS or a process-manager latency benchmark.
Inspect your own callback workload and process transport before choosing caps.

The first increment does not add HTTP-client trace options, native execution,
automatic semantic extraction, payload snapshots, replay or package publication.
The optional private service stays disabled by default for decision recording.
