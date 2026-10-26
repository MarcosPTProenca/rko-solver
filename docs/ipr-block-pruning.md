# IPR unchanged-block evaluation pruning

IPR now omits decoder/objective calls when assigning a guide block would leave the
current chromosome's representation unchanged. It reuses the last completed cost
of that current chromosome. This is local path bookkeeping, not a persistent
fitness cache or phenotype-based deduplication.

## Preserve path decisions, not redundant calls

The original block order, `max(1,num_blocks-1)` **virtual** steps, fixed-block
bookkeeping and strict first-minimum tie decisions remain. An unchanged move still
participates in selection at the known current cost and can consume a virtual step.

It would be incorrect to simply delete equal blocks and shorten the loop to
`different_blocks-1`: that can change which meaningful candidates are explored,
especially when all remaining genuine moves are worse or tied. Shared blocks also
allow the existing path to reach the full guide early; that behavior is retained.

Only same-dtype, byte-identical blocks qualify. Signed-zero changes remain ordinary
evaluations because a decoder can distinguish their signs. Different dtypes are
handled conservatively through the existing assignment/callback path, even if a
cast happens to leave numerical values unchanged. Partial final blocks and
one-coordinate chromosomes remain covered. Caller parents remain detached.

## Exact callback evidence

For the maintained 20-coordinate example, parents differ only in the first
two-coordinate block and the guide improves a sum-of-squares objective:

| | Before | After |
| --- | ---: | ---: |
| Objective evaluations | 55 | **2** |
| Distinct evaluated vectors | 2 | 2 |
| Start evaluations | 10 | 1 |
| Guide evaluations | 45 | 1 |
| Returned chromosome | Guide | Same guide |

An entirely representation-identical pair passed directly to the kernel evaluates
the initial point once. Normal `solve` already backs off before relinking identical
effective parents. A worse/tied genuine last-block move may still be evaluated
repeatedly across virtual steps: removing those further repeats is not this change.

Independent scalar/list oracles run both legacy and pruned policies over identical,
first/last/multiple changed blocks, better/worse/flat objectives and small/partial
dimensions. For pure deterministic scoring they retain legacy result bits and
costs, while actual callback traces exactly match the pruned oracle. Additional
checks cover signed zero, mixed dtypes, original callback failures, an expired
session before the next changed candidate, owned parent snapshots and the scalar
copy-kernel benchmark control.

`benchmarks/ipr-pruned-blocks-reference.jsonl` repeats the bounded 36-case matrix at
1,024 **completed objective evaluations** each, charging all initialization. Pools
remain unique and identical-parent selections remain zero. Pruning lets a fixed
callback budget cover different numbers of paths and consume different RNG draws;
this matrix is not an equal-path comparison or a timing measurement. Earlier
pre-admission and unique-admission reference files remain unchanged historical data.

## Callback and migration contract

**Stateful or stochastic callbacks now run fewer times within a path.** Reusing a
known no-op cost does not reproduce fresh independent observations. Such objective
trajectories and even results can change. No equivalence is promised for callbacks
that depend on invocation count, time, mutation, random state or array identity.
All changed candidates still pass through ordinary decoder/objective isolation,
finite-result validation and cooperative deadline/error handling. No user callback
is interrupted while executing.

Even for deterministic objectives, fixed-time/fixed-evaluation full searches can
change trajectories because saved callbacks admit more path work. Pin `1a17524` to
reproduce the preceding unpruned policy; no legacy flag or new dependency was added.
The internal `ipr-unchanged-block-pruning-v1` replay token marks old IPR incompatible
with RNG-only replay, rather than mislabeling intentional work changes as isolation
regressions.

`benchmarks/ipr.py::ScalarIPR` now shares current pruning/virtual steps, varying only
per-gene versus slice assignment. Do not compare an unpruned scalar kernel with a
pruned solver and call the difference a block-copy speedup. Historical speed data
retains its original source/policy scope; it is not a new measurement of this code.

This establishes reduced redundant callbacks in controlled paths, **not measured
whole-solver throughput, quality, feasibility, CPU/RSS/energy savings or C++ parity**.
No population-size default, native backend or active Knapsack benchmark was changed.
