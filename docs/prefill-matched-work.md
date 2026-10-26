# Opt-in prefill: bounded matched-work pilot

## Question and setup

Does shared startup help when its evaluations are paid from the **same total
objective budget**, rather than added for free before a full search allowance?

`benchmarks/prefill_work.py` exercises real public sequential RKO with prefill false
and true, using the same search seed/parameters and 256 completed objective calls
**including startup**. The pilot covers IPR, GRASP and MultiStart; sphere,
Rastrigin-style rugged, stable-permutation ordering and flat objectives; dimension
20; capacity 10; five paired seeds (200–204). These are 60 pairs / 120 trials,
not application or statistical-superiority evidence.

Identified reference: `benchmarks/prefill-work-reference.jsonl`, runtime source
`41cda1b50f01bffe562a5040121444742b93000e`, Python 3.14.8 / NumPy 2.4.3.
The header records production/config/operator and probe SHA256 values. No production
algorithm, default, native backend or protected benchmark was changed for this study.

## Controls and accounting

- Offline `control=0`; algorithm policy elapsed time is frozen at zero, including
  GRASP's time-dependent acceptance. A real 60-second emergency guard remains;
  trials must exhaust the evaluation cap or fail, never silently accept a timeout.
- Default and prefilled searches must have equal actual search seeds and identical
  first evaluated search chromosomes. Prefilled trials share the same first P raw
  seed vectors across workloads/algorithms for a given seed. Tie mutations can then
  differ by objective, as intended. This is not an equal-path experiment.
- `startup_evaluations + search_evaluations = result.n_evaluations = 256` in every
  row. Startup uses 10–19 observations, leaving 237–246 for the search, whereas
  default initialization/bootstrapping is included in its 256 search observations.
- The private cooperative stop signal rejects the next objective attempt. The
  preceding decoder has already executed: **all trials make 257 decoder calls**,
  of which one is unscored. This diagnostic seam is not a new public evaluation-cap
  API. Candidate work and the rejected attempt are counted explicitly, not timed.
- Every result must equal the best completed observation across both phases.
  Recorded curves, gaps to known optima, full-trace/result hashes and the best
  startup observation support independent inspection.
- Two complete local runs produced **byte-identical JSONL** in the identified
  environment. Unit checks also repeat real trials, verify restored caller RNGs,
  single-gene behavior, bounds and cross-family seed controls.

There are no wall-time, CPU, evaluations/second, memory or energy measurements.

## Results

Counts below are **prefilled better / tied / worse**, minimizing the objective.
Objective units differ between families; do not average their raw costs together.

| Algorithm | Sphere | Rugged | Ordering | Flat |
| --- | --- | --- | --- | --- |
| IPR | 5 / 0 / 0 | 5 / 0 / 0 | 5 / 0 / 0 | 0 / 5 / 0 |
| GRASP | 1 / 4 / 0 | 0 / 3 / 2 | 0 / 4 / 1 | 0 / 5 / 0 |
| MultiStart | 0 / 5 / 0 | 1 / 4 / 0 | 0 / 5 / 0 | 0 / 5 / 0 |

IPR median costs changed from 5.2924 to 2.3454 (sphere), 327.7430 to 195.6588
(rugged), and 1626 to 1467 (ordering). These are within-family pilot summaries,
not percentages or general effect-size estimates.

Did search merely retain a lucky startup vector? In prefilled non-flat trials,
final search results improved beyond the startup pool's best in **15/15 IPR**,
**13/15 GRASP**, and **14/15 MultiStart** cases. Thus IPR's observations are not
explained solely by returning the best initial seed. This does not isolate every
cause: initial material, parent selection, local-search context and the reduced
remaining search budget change together.

## Decision and next evidence

**Keep prefill opt-in.** IPR is promising in this small fixed-work setting, but
GRASP has regressions and most MultiStart pairs tie. Five seeds, one dimension,
one small budget, synthetic objectives and a frozen offline clock cannot justify
a global default, adaptive-policy recommendation or a feasibility guarantee.
GRASP at dimension 20 also exercises its small-unfixed-set fallback; results must
not be extrapolated to the 500-coordinate benchmark regime. The later
[stage-work check](prefill-expanded-work.md#why-the-grasp-wins-need-a-stage-qualification)
shows that even this pilot's 256-call budget cannot finish the first construction
(minimum 350 at packaged settings). These are valid best-observation outcomes,
not evidence about completed construction/refinement cycles.

Next compare wider dimensions, budgets and seeds, then adaptive/timed production
searches and relevant instance families after the protected run finishes. Preserve
startup accounting throughout. Profile callback cost versus framework work before
selecting any Rust/native kernel; this pilot measures no throughput and makes no
C++ quality/trajectory parity claim.

## Hosted expansion

The [completed hosted expansion](prefill-expanded-work.md) retains all paired
observations and qualifies the large-GRASP incomplete-stage limitation. Its manual
`.github/workflows/prefill-study.yml` keeps this pilot's controls while
expanding to dimensions 20/100/500, budgets 256/1024, capacity 10 and seeds 200–219.
Six Ubuntu shards run the original sequential runner, at most three jobs concurrently,
with launch-configured OpenBLAS/OMP/MKL thread limits of one. This is 1,440 pairs /
2,880 trials; none runs on the machine hosting the protected benchmark. Real-time
or adaptive-control evidence is still outside this study.

```bash
gh workflow run prefill-study.yml --ref main
```

The final job validates provenance, complete/nonoverlapping pairs, exact work and
seed controls using `benchmarks/prefill_report.py`. It recomputes per-family
outcomes and medians from trial rows. The report artifact contains compact paired
CSV and summary/provenance JSON; six separate raw-trace artifacts are retained for
90 days. Persist the compact pairs/summary as an identified reference before
artifact expiry; those rows suffice to recompute quality summaries without storing
all full trace records in Git. Input SHA256 values and the workflow URL identify
the raw shards.

The expanded seeds include the original five pilot seeds. Do not merge both data
sets and count overlapping cases as independent observations; the report compiler
rejects duplicate pair keys. Do not average raw objective costs across families.
Expanded outcome claims require completed, validated artifacts, not just a launched
workflow or a passing pilot test.

Reproduce the original pilot (bounded serial work, no timings):

```bash
uv run --locked python benchmarks/prefill_work.py > /tmp/prefill-work.jsonl
uv run --locked pytest src/rko_solver/tests/rko/test_prefill_work.py -q
```

The provenance header changes with checkout/runtime. Compare compatible trial
records rather than expecting a later checkout's whole file to match the archive.

The pinned treehouse benchmark remains at `7ee93ca`; its source, cache, CSV and run
must not be mixed with this corrected-revision pilot or restarted/duplicated.
