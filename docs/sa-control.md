# Adaptive simulated annealing

## Behavior change

Configured **SA now honors `control=1`**. Previously its online TOML grid was read
but not passed to SA, and result metadata reported effective control zero.
Since the packaged global configuration requests online control, this changes
normal configured SA runs—not just direct constructor calls.

Use **`control=0`** to retain the legacy fixed-parameter algorithm. Its proposal,
acceptance, cooling, refinement and RNG sequence remain protected by a scalar
candidate-trace regression. Standalone `SA(...)` still defaults to control zero.
No new runtime dependency or global native/RNG setting is introduced.

```python
solver = RKO(decoder, objective, dimension, algorithms=["SA"], control=1,
             random_state=42)
result = solver.solve(30.0)
print(result.searches[0].control, result.searches[0].parameter_grid)
```

This assumes the usual imported `RKO` and your own decoder/objective/dimension.
As with other adaptive algorithms, grids remain configured through named TOML
axes; nonempty programmatic scalar overrides still require `control=0`.

## Parameter transitions

The named SA axis order is:

1. `sa_max`: integral proposals per temperature level.
2. `t0`: starting temperature scale.
3. `alpha`: geometric cooling factor, strictly between zero and one.
4. `beta_min`: lower shaking intensity.
5. `beta_max`: upper shaking intensity.

The existing validator checks fixed settings, each grid scalar and joint shaking
bounds. Reversed beta combinations are filtered, rather than truncating values
or allowing an invalid state to reach an operator. No feasible states is an error.

A real feasible initial Q state supplies the first schedule's parameters. Before
each temperature level, an epsilon-greedy distance-one action chooses the settings
for that level. `sa_max`, `alpha` and both beta bounds affect actual proposal work.
Selecting a new `t0` does **not** reset a running temperature: the latest selected
scale is used when the next outer cooling schedule starts, reannealing at `0.3*t0`.
Consequently a short search that never completes a schedule may not use a newly
selected t0. This is delayed schedule control, not a promise of immediate thermal
restarts or better learning efficiency.

After proposals and local refinement, a completed level receives reward one for a
new global incumbent, otherwise the shared sign-safe relative penalty against its
best observed proposal. The Q state advances through the chosen action. A level
interrupted before any proposal receives no reward/update; callback exceptions and
cooperative deadline recovery retain the existing SearchSession behavior.

## Reference differences and evidence

This closes missing adaptive functionality, **not complete C++ parity**. The pinned
C++ SA reads online settings and updates Q values, but initializes a random
parameter state while leaving its Q source index at zero, and never applies
selected t0 actions after initialization. Python uses its actual selected initial
state and gives later t0 selections the documented next-schedule meaning. Rewards
use the already-corrected shared controller, including negative objectives/current
Q maxima, and credit local refinement in the completed level. Python's existing
local-search/current-solution continuation remains unchanged from offline mode;
it is not silently rewritten to match the C++ driver's different transition.

Deterministic tests exercise actual Q updates, admissible actions, selected shaking
bounds/proposal counts, both initial states, reannealing temperatures, local-search
credit, empty-level stopping, feasible-grid filtering and offline candidate/RNG
compatibility. Installed smoke exercises offline/adaptive SA in sequential/thread/spawn.

There is **no new timed quality or throughput claim**. Q construction/action work
uses search time like the other adaptive algorithms. The ongoing paired benchmark
must remain pinned and finish unchanged; any later comparison must identify this
new revision and distinguish its active SA control from the older offline fallback.
Match initialization and worker budgets too; cross-revision portfolio differences
cannot all be attributed to this SA correction.
The RNG replay tool reports adaptive cases absent in an older reference constructor
as `not_comparable`, instead of falsely comparing an online run with an offline one.

See [C++ contract audit](cpp-compatibility.md) for remaining GRASP/global
initialization differences and [current limitations](status.md) for release gaps.
