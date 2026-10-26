# GRASP alpha: retain the reference compatibility axis

## Policy

Retain the existing three-axis adaptive grid `[alpha_grasp, hs, he]`, including
packaged defaults, validation, saved-grid schema and result metadata. Do not add
an RCL, remove the axis or emit a deprecation warning: none follows from the
pinned C++ reference. This decision changes no search behavior or replay token.

## Evidence from the reference

In `RKO_Cpp_v2.0` revision `126e35873e50dc308795b58b8c75d37f8de4c857`,
`Program/src/MH/GRASP.h`:

- Lines 213–217 read **three** parameter axes.
- Online `CreateStates` receives all three axes; lines 243 and 270 select alpha
  from coordinate zero of the chosen state.
- `ConstrutiveGreedyRandomized` accepts alpha at line 65 but never uses it in its
  body. Candidate subsets use the 10-percent/small-set rule, line searches sample
  a logarithmic quota and the first minimum wins; alpha does not define an RCL.
- Offline line 285 draws a fresh unused alpha before construction. Python does
  not reproduce that extra draw. Random engines already differ; this policy is
  not a claim of exact reference RNG consumption or seeded trajectory parity.

The Python constructor and packaged TOML preserve the three-axis online shape.
Offline Python settings remain `hs` and `he`; there is no new offline alpha knob.
See [the broader compatibility table](cpp-compatibility.md).

## Not a globally inert parameter

Alpha does not directly change construction at fixed `h`, starting chromosome and
RNG state. However, multiple alpha labels create additional Q states and actions,
including transitions that change only alpha. Q-value initialization consumes
random draws for those actions; action selection can change how and when other
parameters are visited. Removing the axis or collapsing its values therefore
changes adaptive search, even though the constructive kernel never reads alpha.

For example, an explicitly supplied grid
`[[0.1, 0.9], [0.5, 0.25], [0.01, 0.001]]` has eight states. Replacing only its
first axis with `[0.1]` leaves four. With distinct values and all states feasible,
the current controller includes self-actions, so each state's action count changes
from four to three. This is a graph-size example, not a memory/speed/quality result.

Existing three-axis grids continue to load and save unchanged. Users who
intentionally want a smaller controller can already supply a singleton first axis
in their custom `grasp.toml`, retaining the required name:

```toml
[online]
alpha_grasp = [0.1]
hs = [0.5, 0.25]
he = [0.01, 0.001]
```

Keep the existing `[offline]` section when editing a copied parameter file.
This is an opt-in **different adaptive policy**, not a trajectory-preserving
optimization or a recommended replacement for the packaged defaults. Direct
constructors likewise accept `parameters=[[0.1], [0.5, 0.25], [0.01, 0.001]]` with
`control=1`. No migration is required for existing configurations.

## Regression coverage and limits

Tests compare actual construction callbacks, result bits/costs and final owned
RNG states for different singleton alpha labels. Separate checks use the real
packaged grid and controller to establish alpha-only actions, graph cardinalities
and changed initialization RNG states when its axis is collapsed.

These distinguish a kernel-unused parameter from an adaptive-policy coordinate;
they do not establish optimization quality, throughput, memory savings, full
C++ parity or release readiness. Full-driver population initialization and
matched-work comparisons remain separate work.
