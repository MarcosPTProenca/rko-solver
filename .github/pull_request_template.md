<!--
Title: `[L1|L2|L3] <type>(<scope>): <summary>`. The level is the larger of complexity and risk:
L1 trivial and local, L2 moderate with isolated new behavior, L3 broad, risky or breaking.
Replace every placeholder; delete lines that do not apply. State facts you observed, not intentions.
-->

## Why

<!-- The problem, why this approach, and the alternative you rejected if it matters. -->

## Summary

<!-- The smallest view that explains the change: schematic diff, file tree, pseudocode or diagram. -->

## Evidence

- **Before:** <!-- Observed output or failing test that shows the problem; say so if not applicable. -->
- **After:** <!-- Comparable observed result. -->

## Tests executed

| Type | Scope | Command | Observed result |
| --- | --- | --- | --- |
| | | | |

<!--
Record RED before the change and GREEN after it. Prefer end-to-end flows through real entry points.
Mark everything you did not run as NOT RUN and say why. Never report a result you did not observe.
-->

## How to validate

```bash
```

## Merge risk

**Reversal:** <!-- Reversible (two-way door) or hard to reverse (one-way door), and how to revert. -->

**Blast radius:** <!-- Who or what can be affected beyond the edited files, and what is not covered. -->

## Checklist

- [ ] The description explains why, not only what
- [ ] No credentials, tokens or private data are committed
- [ ] The title carries `[L1]`, `[L2]` or `[L3]`
- [ ] Tests list real commands and results; gaps are marked NOT RUN
- [ ] Documentation and the changelog match the new behavior
