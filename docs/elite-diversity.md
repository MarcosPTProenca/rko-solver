# Chromosome-unique elite pools

`SolutionPool` retains at most one entry per coordinate-equal chromosome. This rule
is applied **inside the existing insertion lock**, including manager-hosted pools.
No proxy API, public return shape, dependency or configuration flag was added.

## Admission and repeated observations

- **Different chromosomes with tied objectives remain eligible.** This is not
  objective-value deduplication, unlike the audited C++ update rule.
- Equal or worse observations of a stored chromosome are ignored.
- A strictly better observation replaces that chromosome's stored cost, source and
  owned keys, then restores ordinary ascending-cost ordering. It does not evict a
  different elite just to retain another copy of the same vector.
- Novel chromosomes retain the existing capacity/worst-cost rule and stable ties.
- Validation still happens before duplicate detection, including for invalid
  repeated observations. Inserted and returned arrays remain detached.

Equality uses the original finite coordinate values, without float packing,
rounding, tolerance or objective-based keys. Adjacent representable values, different
large integers and different lengths remain distinct; numerically equal integers/
floats and signed zeros compare equal. Original scalar representation is retained.

This is **not a fitness cache**. Decoder/objective callbacks still execute and count
normally. If a stochastic callback reports a better cost at the same chromosome,
that improved observation remains usable; the pool keeps its best observation,
not an average or uncertainty model. Callers remain responsible for their objective
semantics and stochastic state.

## IPR and custom pools

Stored uniqueness prevents ordinary local/managed IPR from sampling two copies of
one chromosome. IPR also checks the converted binary64 parent arrays before
relinking: if they are equal, it uses the existing bounded sleep/backoff instead.
That covers custom providers returning clones and distinct large integers collapsing
to the same effective search coordinates. Custom pools still need two usable
parents to support search; they are not silently rewritten or endlessly reseeded.

The existing startup rule still attempts **at most one** extra seed. If both random
draws are identical and only one elite survives, IPR backs off within its ordinary
stop/cancellation checks. No unbounded “retry until distinct” loop is introduced.

## Validation and work evidence

Local tests exercise original failing clone admission, valid tied vectors, improved/
worse observations, detached inputs, exact numerical identity, invalid duplicates
and concurrent writers. Manager tests exercise authoritative RPC admission and
pair retrieval. IPR tests cover custom effective clones and repeated seed draws;
its prefilled NumPy initialization order remains unchanged, while later pair-draw
traces reflect the corrected population occupancy.

`benchmarks/references/ipr-unique-elites-reference.jsonl` repeats the bounded 36-case diagnostic
from [the historical diagnosis](ipr-diagnosis.md), at 1,024 completed evaluations
per case, with all initialization charged. **All 36 cases have unique retained
vectors and zero identical-parent selections.** Production traces match the
isolated exact-vector admission control in those deterministic workloads.
The older `references/ipr-diagnosis-reference.jsonl` remains the identified pre-fix baseline.

The two-seed scores remain unchanged in that matrix; ten-seed scores change in
mixed directions relative to old production admission. **No general quality,
feasibility or end-to-end speed improvement is established.** A subsequent
[unchanged-block correction](ipr-block-pruning.md) now reduces the controlled
one-differing-block kernel from 55 to two callbacks while retaining virtual path
and deterministic tie behavior. Stochastic callback trajectories can change.

## Migration and scope

This is a behavioral change for **every algorithm sharing `SolutionPool`**, not
only IPR. Occupancy counts, guides, sampled ranks, pool-aware refinement and seeded/
timed trajectories can change. Query/serialization/proxy schemas remain unchanged.
Pin `0c3a573` to reproduce the preceding duplicate-admitting policy; no legacy flag.
The running Knapsack comparison pinned to `7ee93ca` is not changed or reinterpreted.

The RNG-isolation replay tool deliberately supplies the **current pool to both
algorithm implementations**. Such replay checks compare RNG ownership within the
same pool context; they do not reproduce a historical application's old admission
policy. Historical replay/performance files retain their original source scope.
