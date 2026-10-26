# C++ reference and compatibility

Behavioral reference: [RKO_Cpp_v2.0](https://github.com/RKO-solver/RKO_Cpp_v2.0),
revision `126e35873e50dc308795b58b8c75d37f8de4c857`. Compare executable behavior,
not only comments or algorithm names. Full solver equivalence is **not proven**.

## Checked neighborhood contracts

The reference `Program/src/Method.h` calls strategy 1 "first improvement", but its
immediate `return` statements are commented out. `SwapLS`, `InvertLS` and `FareyLS`
therefore continue the sweep after accepting an improvement. All three shuffle
`RKorder` unconditionally. Python already followed that transition policy;
changing it to immediate return would diverge from this reference.

| Setting | Reference behavior preserved in Python |
| --- | --- |
| strategy 1 | Sequential acceptance: next candidate starts from the best accepted vector; finish the sweep |
| strategy 2 | Evaluate all neighbors from the fixed input; return the best improvement |
| supplied order | Base indices are reshuffled, not treated as a fixed exploration schedule |

Six deterministic cases (three operators, both strategies) were verified by
compiling the unmodified C++ function bodies with minimal solution/data types,
a linear decoder and controlled exploration order/Farey midpoint draws.
The same outputs and full candidate counts are regression-tested in
`src/rko_solver/tests/components/test_cpp_operator_contracts.py`. This was a kernel
comparison, not a build/test of the complete C++ application or random generator.
The normalized reference `Method.h` SHA256 is
`0f202c543c424a77db3f44220986ee956b3b17c0900a2454fcb122f25339262d`.

Python returns new vectors and copies the supplied index list instead of
mutating caller-owned input as C++ does. Random streams and repeated-call index
state are not identical; these tests compare matched deterministic transitions.
The portfolio currently uses strategy 2 defaults; a run-level strategy setting
is not yet exposed. Operator docstrings describe actual sweep semantics.

## Deliberate corrections, not unchanged trajectories

Python now keeps selected GA fitness with its chromosome, recomputes current
Q maxima after updates, handles negative/extreme rewards consistently, filters
infeasible joint parameter states, covers partial IPR blocks and retains superior
initial simplex vertices. Some issues were inherited from the reference.
These are explicit correctness choices; do not silently revert them just to
match a historical defect or claim byte-for-byte C++ search trajectories.

## IPR initialization contract

The C++ driver calls `CreatePoolSolutions` before invoking its selected searches;
standalone IPR therefore starts with an elite population. Python previously started
with an empty pool and could spend its entire allocation waiting, returning only
its first random candidate. This is an initialization difference, not evidence
that compiling the same waiting loop would improve search quality.

Python IPR now inserts its initial evaluated candidate and tries at most one extra
seed if fewer than two elites exist at startup. Both evaluations use ordinary
callbacks, assigned RNGs and the existing search deadline. Prefilled pools retain
their initialization draw order. Initialization is bounded even when a custom pool
or capacity-one pool cannot retain a pair; existing backoff then applies.

This restores usable standalone relinking, **not full-driver parity**: C++ seeds
its configured pool population, whereas Python only bootstraps the missing pair
locally in IPR. A [fixed-work diagnosis](ipr-diagnosis.md) exposed clone collapse
and redundant block evaluations. Stock Python pools now enforce
[chromosome uniqueness](elite-diversity.md), retaining different tied-cost vectors;
IPR skips identical effective parents. C++ update rejects objective duplicates,
which is a different policy. Python's [unchanged-block pruning](ipr-block-pruning.md)
reuses known current costs for representation-identical moves while retaining
virtual path/tie decisions. This intentionally changes callback counts, not an
assertion of matching C++ evaluation schedules or stochastic trajectories.
Global initialization, random engines, block-path transitions and
throughput still require separate matched comparisons. Random seeds are not generic
feasibility repairs; no C++-matching quality claim follows from this correction.

## SA adaptive and GRASP construction audit

SA now implements online actions rather than silently falling back to fixed
settings. [The SA control contract](sa-control.md) describes actual parameter
application and migration to `control=0` for the legacy path. Python uses the
sampled initial Q state, credits completed levels including refinement with the
shared sign-safe reward, and applies selected t0 to the next schedule. C++ leaves
its initial Q source at zero and never applies later t0 selections. These are
explicit corrections, not claims of identical trajectories or scientific quality.
Python's legacy refinement/current-incumbent transition is retained; full SA
transition parity remains unproven.

GRASP's [line-candidate correction](grasp-line-search.md) now restores the nominal
reference lattice, sample quota and optional incumbent inclusion. Exact-ratio bounds
and sparse sampling define safe behavior beyond the reference's float/int limits.
Construction iteration/subset counts now also match the nominal reference rules.
Other audited construction/refinement policies remain separate:

| Contract | Pinned C++ | Current Python |
| --- | --- | --- |
| Line candidate universe | Bounded lattice around the gene, then randomized subset | Same nominal lattice with exact-ratio bounds; no dense allocation |
| Candidate evaluation count | At most `ceil(log2(int(1/h)))+1` sampled lattice values | Matching nominal quota, safely extended to tiny/large steps |
| Incumbent in subset | May be omitted; construction can worsen | May be omitted; global solver best remains protected |
| Small unfixed sets | Search every remaining coordinate when the 10% subset is below two | Same fallback, recomputed after each commit |
| Construction intensity | Integer loop against a real bound (effectively ceiling) | `ceil(n*intensity)`, including one construction step for n=1 |
| Shuffle ordering | Shuffle a copy of unfixed coordinates | Shuffle a copy; unfixed order survives every pass (aligned) |
| Improving fine-grid step | Keep the current h | Keep the current h; previous reset to hs removed |
| Initial refinement | Score the initial vector before construction | Score the initial vector, then insert it; no initial RVND |
| Alpha parameter | Read/selected but unused by construction | Retained adaptive state axis; no direct construction effect ([policy](grasp-alpha.md)) |

The former near-only Python grid could miss available candidates while making
more callback evaluations. Bounded dyadic oracle cases now cover the wider universe,
ordered samples and callback budget; this is not a full compiled-C++ quality test or
bitstream/rounding equivalence. The unfixed-order shuffle now also matches the
reference copy semantics. Alpha is deliberately retained for reference state-graph
compatibility, not interpreted as an RCL. Its indirect adaptive effects and the
remaining offline RNG-draw difference are documented in [the alpha policy](grasp-alpha.md). The ongoing paired
benchmark remains unchanged; full validation is hosted to avoid local contention.

## Full-driver initialization

The [startup contract audit](driver-initialization.md) separates shared-pool
seeding from private algorithm populations. C++ fills its shared pool before
launching searches and makes one cost-tie perturbation pass; Python starts empty
by default and treats `pool_size` as capacity. An explicit `prefill_pool=True`
now enables the reference-style fill/tie pass with isolated callbacks, separate
startup reports and fresh search allowances. Coordinate-unique admission and RNG
engines remain different. Equal nominal capacities/durations still do not imply
exact startup work, diversity or trajectory parity.

## BRKGA-CS undefined correlation

Community construction now excludes constant chromosomes and chromosomes with
fewer than two observations. Pearson correlation is undefined in those cases;
there is no community edge, including when floating-point rounding previously
fabricated a finite coefficient for a constant array. Nonconstant pairs whose
normalization fails numerically (for example, covariance underflow) also add no
edge. No epsilon cutoff is used: finite small-variance correlations remain valid.

Python still calls NumPy for defined coefficients. A fresh local `np.errstate`
turns invalid/divide faults into handled `FloatingPointError`, not ignored warnings;
the caller's floating error policy is restored. Label propagation, representative
selection and their RNG draws are unchanged for the same graph. Direct regressions
cover the real warning, one-gene inputs, constants, small variance, error-policy
restoration and a hand-worked two-component graph. Defined graph/RNG results were
also checked against the unmodified prior Python source. The diagnostic replay
contract is `brkga-cs-defined-correlation-v1`, not a new user option.

This deliberately removes spurious constant-array edges and can change those
search trajectories; it is not an exact replay or quality-improvement promise.
The pinned C++ `Program/src/MH/BRKGA_CS.h::PearsonCorrelation` has an unchecked
raw-moment denominator. Source inspection shows an ordinary zero-denominator NaN
fails its graph threshold, but no compiled whole-solver equivalence follows.
Python retains its existing **absolute** correlation and inclusive `>= sigma`
rule; the reference uses signed `pearson > sigma`. This pre-existing difference
is not silently changed by the numerical-warning fix. Full BRKGA-CS parity and
current multi-instance quality remain unproven.

See NumPy's [coefficient/rounding contract](https://numpy.org/doc/2.4/reference/generated/numpy.corrcoef.html)
and [scoped floating error handling](https://numpy.org/doc/2.4/reference/generated/numpy.errstate.html).

## Still to resolve

- GRASP offline unused-alpha RNG draw differs; exact random-engine/trajectory parity is not claimed.
- Full SA/GRASP transition and refinement comparisons beyond adaptive plumbing.
- Full per-algorithm initialization, transition, restart and stopping contracts.
- Full-driver population initialization and comparable timing/work accounting;
  owned search RNGs and CPU-bounded remaining-time process scheduling are implemented.
- Mapping additional C++ run settings into actual Python functionality rather
  than accepting unsupported fields. Current terminal/configuration usage is in
  [running.md](running.md).
- Matched-work, multi-instance measurements with strong optimization baselines.

These remain [release limitations](status.md). No competitive-quality or
whole-solver parity claim follows from six neighborhood examples.
