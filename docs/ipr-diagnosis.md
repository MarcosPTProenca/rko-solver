# IPR fixed-work diagnosis

**Historical baseline:** the findings below describe `0c3a573`, before production
chromosome uniqueness. The [elite-diversity correction](elite-diversity.md) now
prevents duplicate admission and skips identical effective parents. The original
reference remains unchanged; separate post-fix work records are maintained.
[Unchanged-block pruning](ipr-block-pruning.md) subsequently reduces the direct
one-differing-block case from 55 to two evaluations, retaining virtual decisions.

The empty-pool bootstrap enables standalone IPR to start relinking. It does not
establish adequate population diversity, efficient path work or solution quality.
This diagnostic isolates those questions without another timed quality benchmark.

## Reproduce

From the repository root:

```bash
uv run --locked python benchmarks/ipr_diagnosis.py
```

Defaults: 32 coordinates, seeds 200–202, sphere/threshold/flat objectives and **1,024
completed objective evaluations per case**. The 36-case matrix crosses:

- Ordinary two-seed bootstrap versus ten externally initialized elites.
- Production pool admission versus a **diagnostic-only exact-vector duplicate
  rejection control**.

Every external initialization evaluation is charged. The ten-elite arm also pays
for production IPR's unconditional initial draw, so its split is **11 seed + 1,013
path evaluations**, versus **2 + 1,022** for ordinary bootstrap. First-two-seed
fingerprints match across all four arms. A final path may be interrupted by the
cap; `pairs_started` includes an entered but incomplete path. SearchSession still
retains the best completed candidate and restores callbacks.

The exact-vector control is isolated and single-threaded; it is **not a production
pool implementation or API option**. It retains different chromosomes with tied
costs. Do not substitute objective-value equality for genotype equality.

`benchmarks/ipr-diagnosis-reference.jsonl` records the source revision, solver/probe
source SHA256, Python/NumPy versions, one direct path case and all 36 count traces.
No wall/CPU timings are collected. The public C++/Python Knapsack benchmark,
worktree, cache, environment and output are not accessed by this script.

## Findings on this matrix

### 1. Pair formation works, but occupancy is not diversity

All ordinary two-seed runs perform actual path evaluations, including genuine
intermediate vectors. Nevertheless, all three sphere runs end with **ten pool
entries containing only one distinct vector**. Production pool admission accepts
exact clones when capacity remains or their score beats the worst entry; IPR can
reinsert an unchanged best path/start repeatedly.

Across the three ordinary-bootstrap seeds:

| Objective | Distinct final vectors / ten entries | Identical-parent pairs / 48 started |
| --- | --- | --- |
| Sphere | 1, 1, 1 | 25 / 48 |
| Threshold | 2, 2, 2 | 13 / 48 |
| Flat | 2, 2, 2 | 24 / 48 |

The exact-vector control eliminates identical-parent selections in this matrix.
**It does not improve the best objective in any of the nine two-seed cases.**
Removing wasted work is not by itself evidence of improved quality.

### 2. Unchanged blocks are evaluated repeatedly

An independent direct production-kernel case has 20 coordinates and two distinct
parents differing in exactly the first two-coordinate block. The relinker makes
**55 objective calls for only two distinct vectors**: ten start evaluations and
45 guide evaluations. No genuine intermediate vector exists for that pair.

The implementation visits every initially unfixed block, even when its values
already equal the guide. It takes `num_blocks-1` steps, not a count based on actual
parent differences. Therefore shared blocks can also make the full guide reachable
early; “always stops before the guide” is not a valid invariant.

### 3. More starting diversity helps these objectives, not universally

At the same total evaluation cap, ten initial elites improve the final sphere and
threshold scores over ordinary bootstrap in all six numerical comparisons here;
flat scores remain tied. Production identical-parent selections fall to 3/144
started pairs across those nine larger-population cases, but some pools still lose
diversity. The exact-vector control combined with ten seeds gives **mixed** results
relative to the production ten-seed arm: it can finish better, equal or worse.

All 36 traces produce **zero path candidates outside the seed coordinate alleles**.
Internal relinking only copies seed-derived blocks; it cannot invent new coordinate
values. A larger seed population changes the available search space, not merely
startup overhead. These small separable synthetic objectives are deliberately
sensitive to that restriction; they are not representative all-domain evidence.

## Reference-source audit

Read-only reference: `RKO_Cpp_v2.0` revision
`126e35873e50dc308795b58b8c75d37f8de4c857`.

- `Program/src/MH/IPR.h` sets `direction=1` every outer iteration. Its external
  branch exists but is inactive; missing external search is **not** an established
  explanation of this comparison.
- `Program/src/Method.h::UpdatePoolSolutions` rejects already-present **objective
  values**, while the historical Python policy admitted exact vector clones
  (now corrected separately). Copying that
  policy would also reject distinct tied vectors; it is not an automatic safe fix.
- The C++ driver initializes its population before invoking IPR and perturbs some
  tied-cost initial entries. The ten-random-elite arm here is **not** that complete
  initialization policy or a compiled C++ execution.
- C++ difference flags use coordinate modulo block count while swaps use contiguous
  blocks, and its block count truncates. Python safely covers partial blocks; do
  not reproduce reference indexing/edge defects to claim parity.

## Subsequent corrections and remaining scope

Exact duplicate admission/identical-parent selection is now addressed by the linked
production correction, retaining valid tied vectors and bounded deficient-pair
behavior. The linked unchanged-block correction now reuses known current costs for
representation-identical moves, retaining virtual steps and deterministic path/tie
results. Stochastic callback counts change explicitly. A full-population policy
still requires controlled evidence; this diagnosis does not choose one. The
[full-driver startup contract](driver-initialization.md) details the reference
pool fill, bounded cost-tie pass, default empty startup and distinct clock boundaries.
The opt-in `prefill_pool=True` phase now supplies shared context with explicit
startup accounting; it is not a demonstrated IPR quality/feasibility correction.

The original diagnosis itself changed no production transition/admission policy;
the linked subsequent correction is a separately identified behavior change.
It does **not** establish Knapsack feasibility after the bootstrap, end-to-end
speedup, full C++ parity or a reason to rewrite the core in Rust. The ongoing pinned
benchmark remains a historical baseline and must not be relabeled as current IPR.
