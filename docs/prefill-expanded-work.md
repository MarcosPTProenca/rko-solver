# Expanded prefill study: quality and incomplete-stage limits

## Evidence retained

Hosted workflow [39690019745](https://github.com/MarcosPTProenca/rko-solver/actions/runs/39690019745)
completed six Ubuntu shards and validated their paired controls before aggregation.
Source revision: `831af1096ed0913b1f88da78aa78161aec1d8118`; Python 3.14.8 /
NumPy 2.4.3. Production/config/operator and runner hashes match the original pilot.
The workflow requested OpenBLAS/OMP/MKL thread limits of one at launch; it did not
change library defaults or the local benchmark environment.

- Dimensions: **20, 100, 500**; total budgets: **256, 1,024**; capacity: **10**.
- Seeds **200–219**; IPR, GRASP, MultiStart; sphere, rugged, ordering and flat.
- **1,440 pairs / 2,880 trials**, totaling 1,843,200 completed objective calls.
- Every trial pays startup plus search from the same objective cap, with one
  additional explicitly unscored decoder attempt. Offline control and frozen
  policy clocks remain as in [the pilot methodology](prefill-matched-work.md).
- Actual search seeds and first search chromosomes agree within pairs; raw
  startup vectors agree across families/algorithms/budgets at the same dimension.
- All six artifacts were independently downloaded and revalidated locally; the
  recomputed compact CSV and summary matched the hosted outputs byte for byte
  before the retained CSV's line endings were normalized to LF for Git (values and
  hashes unchanged). The report writer now emits LF too. No expanded trials or
  timing experiments ran on the benchmark machine.

Durable evidence is `benchmarks/references/prefill-expanded-pairs.csv` (all paired costs,
startup evidence and control hashes) and `benchmarks/references/prefill-expanded-summary.json`
(72 family-specific groups, provenance, input SHA256 values and workflow URL).
The six full trace artifacts remain on the workflow for 90 days. Compact evidence
is retained in Git and suffices to recompute the quality summaries after expiry.

The expanded sample **includes** the original five seeds. It is not an independent
1,440-pair sample plus another 60 pilot pairs. All 120 overlapping trial trace
hashes and all 60 paired outcome labels reproduced. Only 94/120 whole records were
bit-identical across environments: some curve/startup/final numerical values differ
in their last bits; maximum final-cost difference was 8.9e-16. We did not isolate
whether hardware, native math or launch configuration caused those differences.

## Outcomes

Below: prefilled **better / tied / worse** across the three non-flat families,
20 pairs per family, hence **60 pairs per row**. This aggregates outcome counts,
not raw objective costs or independent probability estimates. The summary JSON
retains every family separately. Flat objectives account for another 120 tied
pairs per algorithm and are excluded here.

| Algorithm | Variables | Budget | Better | Tied | Worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| IPR | 20 | 256 | 54 | 0 | 6 |
| IPR | 20 | 1,024 | 59 | 0 | 1 |
| IPR | 100 | 256 | 54 | 0 | 6 |
| IPR | 100 | 1,024 | 60 | 0 | 0 |
| IPR | 500 | 256 | 55 | 0 | 5 |
| IPR | 500 | 1,024 | 60 | 0 | 0 |
| GRASP | 20 | 256 | 6 | 37 | 17 |
| GRASP | 20 | 1,024 | 17 | 33 | 10 |
| GRASP | 100 | 256 | 25 | 21 | 14 |
| GRASP | 100 | 1,024 | 0 | 32 | 28 |
| GRASP | 500 | 256 | 53 | 7 | 0 |
| GRASP | 500 | 1,024 | 49 | 9 | 2 |
| MultiStart | 20 | 256 | 3 | 55 | 2 |
| MultiStart | 20 | 1,024 | 0 | 57 | 3 |
| MultiStart | 100 | 256 | 2 | 54 | 4 |
| MultiStart | 100 | 1,024 | 0 | 60 | 0 |
| MultiStart | 500 | 256 | 2 | 57 | 1 |
| MultiStart | 500 | 1,024 | 0 | 59 | 1 |

Across 360 non-flat pairs per algorithm:

- **IPR:** 342 better, 18 worse. Promising across tested settings, not guaranteed.
- **GRASP:** 150 better, 139 tied, 71 worse. Effects depend strongly on regime.
- **MultiStart:** 7 better, 342 tied, 11 worse. Mostly neutral, not universal gain.

Counts use strict finite-cost comparison, as in the pilot. No non-tied difference
was within a 1e-12 scaled numerical-nearness check; the counts are not merely
last-bit perturbations. This is descriptive evidence, not a significance test:
settings and seeds repeat across related cohorts, and objective units differ.

## Why the GRASP wins need a stage qualification

At 500 variables, **all 53 wins at budget 256 and all 49 at budget 1,024** retained
the startup pool's best without improving it afterward. These outcomes do not show
that GRASP's constructive method or RVND improved. Additional initial candidates
can win simply because the baseline's short search has not reached those values.

The packaged offline setting is `hs=0.125`, with four objective calls per sampled
coordinate line search. Construction performs at least `ceil(0.3*n)` commits.
Summing each pass's actual 10-percent/small-set subset rule gives the following
source-derived minimum work **through the first complete construction**, including
one initial and one final score but excluding optional shared startup and RVND:

| Variables | Minimum commits | Coordinate line searches | Minimum objectives |
| --- | ---: | ---: | ---: |
| 20 | 6 | 87 | 350 |
| 100 | 30 | 243 | 974 |
| 500 | 150 | 6,315 | 25,262 |

The lattice at this step has enough eligible positions to fill its four-call quota.
Real randomized intensities can require more work than these lower bounds. Thus
neither 500-variable budget can complete the first construction; even the original
20-variable/256-evaluation pilot stops beforehand. The callback incumbent recovery
still returns the best completed exploratory point as intended. That is a valid
limited-budget API outcome, but not steady-state evidence about complete GRASP
construction/refinement cycles. Increasing dimensions alone did not remove this
methodological limitation.

IPR's prefilled final cost improved beyond startup best in 359/360 non-flat cases.
That separates most of its observed gains from merely returning the best seed,
but does not isolate every parent-selection/path/material effect. More starting
material, pool context and reduced remaining search work still change together.

## Decision and next steps

**Keep prefill opt-in; do not introduce a default or native rewrite.** The expanded
study finds broader IPR benefits but also 18 regressions, mixed GRASP behavior and
mostly neutral MultiStart. No quality/feasibility guarantee follows.

Next instrument diagnostic stage/cycle completion and use GRASP budgets sufficient
for complete construction and refinement, rather than interpreting incomplete-stage
wins as algorithm improvement. Then test adaptive and real-clock behavior plus
relevant instance families. Keep startup accounting in every comparison. Profile
callback versus framework cost before choosing native kernels, after the protected
run finishes.

This study measures no throughput, CPU/RSS/energy savings, Knapsack feasibility,
C++ trajectory/quality parity or release readiness. The protected `7ee93ca` run and
its source/cache/CSV were not changed, restarted, duplicated or merged with these
corrected-revision observations.
