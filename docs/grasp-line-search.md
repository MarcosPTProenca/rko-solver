# GRASP lattice and construction contracts

## Corrected behavior

GRASP now samples the wider coordinate lattice described by the pinned C++
reference instead of exhaustively testing a small logarithmic-radius neighborhood.
This changes **both `control=0` and `control=1`**, including random draw ordering,
callback counts and timed search trajectories. There is no legacy-policy flag;
pin `1bf4ce3` for the old near-only lattice, or `0fac7f5` for the lattice-only
correction before construction counts/subsets changed. SA's separate
`control=0` migration does not restore the old GRASP neighborhood.

The correction is domain-neutral. No decoder, feasibility repair, parameter value,
new dependency or public setting was added. The running C++64/Python benchmark
remains pinned and must finish unchanged; later comparisons need a new identified
source revision rather than mixing old/new rows.

## Candidate and evaluation contract

For the current coordinate `z` and step `h`:

1. Compute `m = max(1, floor(1/h))` and `K = floor(m/2)+1`.
2. The incumbent is one candidate position. Eligible shifted positions are
   `z + tau*h` and `z - tau*h`, for integer `tau` from one through K, restricted
   to `[0,1)`.
3. Uniformly sample `min(eligible_count, ceil(log2(m))+1)` positions **without
   replacement**, in random order, and evaluate only those positions.
4. Return the first sampled minimum, retaining the first sample on objective ties.

The incumbent is **not forced into the subset**. A line search can therefore
return a worse point, as the randomized reference construction permits. Existing
GRASP global-incumbent tracking and Metropolis acceptance remain in charge of
retaining/accepting constructed solutions; the solver still preserves an already
known superior incumbent.

For example, with `z=0.875` and `h=1/1024`, the old implementation evaluated 23
points only in `[0.8642578125,0.8857421875]`. The corrected lattice contains 641
positions and samples 11. The coordinate `0.375` is now eligible; a controlled
sample test reaches it. A random subset need not include that point, so this is
**coverage and callback-budget evidence, not a quality or speed benchmark**.

## Bounded representation and numeric safety

The implementation counts positive/negative positions analytically and maps sampled
ranks back to the interleaved reference order. Sparse swap-removal samples slots
without constructing a list of size `1/h`. Auxiliary lattice storage is O(q), plus
ordinary candidate-vector storage; each score receives a detached binary64 vector.
Callback retention/mutation cannot change caller inputs or later sample contexts.

Bounds and shifted coordinates use exact integer ratios of the supplied binary64
values. This avoids reciprocal overflow and machine-integer limits. Even the
smallest positive binary64 step requires at most **1075 objective calls** for a
complete line search, subject to the existing callback deadline. Inputs must have
a positive finite representable step and a finite varying coordinate in `[0,1]`;
invalid inputs fail before RNG/callback effects. Larger-than-unit steps safely
reduce to available positions rather than computing an undefined logarithm.

An existing endpoint-one incumbent is retained for compatibility with mirror
operators/reference initialization. Shifted positions remain half-open: if an
exact valid position rounds to `1.0`, it is rounded inward to the largest binary64
value below one. At extremely fine precision, distinct lattice positions may
round to equal coordinate values; sampling is without replacement over **positions**,
not unique rounded keys or decoded solutions.

This is not a reproduction of C++'s binary32 step conversions, integer overflow
or random-device bitstream. Bounded dyadic examples match the literal reference
candidate loop and quota. Decimal/tiny steps follow the explicit safe binary64
contract above. All draws use the existing `RandomStreams` scope (or the documented
standalone global fallback), not a new unseeded generator.

## Construction intensity and coordinate subsets

Each construction draws the existing intensity in `[0.3,0.7]` and commits
`ceil(n*intensity)` coordinates, capped by the number of unfixed coordinates.
This matches an integer loop tested against the real bound, rather than
truncating it first. A one-variable problem therefore performs one construction
step instead of zero. Integer bounds retain their existing iteration count.

On **each pass**, compute `k=floor(0.1*remaining)`; if `k<2`, search **all** remaining
coordinates. At 20 remaining coordinates this means two line searches, but at 19
it means nineteen. The subset is recomputed after every commit. Each candidate
sees the same committed partial solution; only the winning coordinate is fixed
and removed. Existing in-place shuffle semantics and first-minimum tie handling
are unchanged, not claims of matching C++ engines or seeded coordinate identities.

These rules apply in both control modes and can increase construction work.
Existing callback deadline/error boundaries and caller ownership are retained.
This is a search-contract correction, **not a demonstrated performance or quality
gain**. The preceding lattice-only revision also has different full-search RNG
trajectories and is not an RNG-isolation replay baseline for this version.

## Validation and remaining differences

Tests compare ordered callback traces/results/private RNG states against an
independent materialized reference lattice, enumerate all 24 ordered three-of-four
subsets, exercise incumbent omission/ties, preserve a solver's superior global
incumbent, and cover boundaries, readonly strided float32 inputs, callback mutation,
exceptions, deadline stopping, tiny/subnormal steps and bounded traced allocation.
These are deterministic contract checks, not a full compiled-C++ application test.
Construction tests independently enumerate real-bound iterations and candidate
contexts, including one-variable progress, adjacent float bounds, the 21/20/19
threshold transition, distinct committed coordinates and callback stopping/failures.
Installed smoke exercises GRASP in sequential/thread/spawn and both control modes.

Historical RNG-only replays of the old GRASP algorithm are now reported as
`not_comparable` for both modes; the constructor carries the
`grasp-construction-v4` compatibility token. The token is diagnostic, not a
user-selectable search policy. Older replay reference files retain their original
scope; they are not evidence for the corrected candidate policy.

Construction now shuffles a fresh copy on every pass and removes only the committed
coordinate from the original unfixed list, preserving its surviving order. Ties
still select the first candidate in each pass's shuffled order. Subset sizes, intensity, line-search
sampling and callback ownership are unchanged. The extra list costs O(remaining)
scratch; no memory or throughput improvement is claimed. Seeded candidate paths
can differ despite the same number of shuffle calls; pin `a388e47` for the previous
in-place policy. RNG engines still differ from C++, so this is not bitstream parity.

The [alpha axis is retained for reference compatibility](grasp-alpha.md): it
influences adaptive state/action geometry, not construction directly. In-place subset shuffling was
replaced by the reference's shuffled-copy semantics. Step decay and initial
handling now follow the reference grid: h is halved only after a construction that
fails to improve the global best, and the initial random vector is scored once and
published without refinement. All are deliberate behavior changes with contract
tests and no measured quality claim. Full-driver initialization, quality,
throughput and native-backend decisions still require matched comparisons. See
[the contract table](cpp-compatibility.md) and [current limitations](status.md).
