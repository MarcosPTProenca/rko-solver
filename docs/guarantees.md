# What RKO guarantees about convergence

[Documentation index](index.md) · [Limitations](status.md) · [Parameter reference](parameters.md)

This page states the one convergence property the library can defend with a proof and a test,
and what it does **not** tell you. It is a safety property, not a speed or quality claim.

## The guarantee

Let the decoder map a uniform random key vector to a solution with objective value at most
`f* + e` (a target you accept) with probability `mu`. We call `mu` the **target mass**: it is a
property of your decoder, not of the algorithm. If the search

1. draws at least one fresh uniform key vector per generation or restart, and
2. never discards the best evaluated solution,

then, for every tolerance `e` whose target has positive mass, the best cost found reaches
`f* + e` almost surely. The best cost therefore converges to `f*` when every `e > 0` has positive
mass, and equals `f*` exactly only if an optimal solution itself has positive mass. After `k` fresh
uniform draws the chance of having missed a target of mass `mu` is at most `(1 - mu)^k`.

For BRKGA the draws are the `p` keys of the initial population plus `m = int(p * pm)` random
mutants in every generation, so after `G` generations

```text
P(miss) <= (1 - mu) ** (p + m * G)
```

The bound ignores crossover and the elite local step, which can only help.

Both conditions hold in the library:

- **Fresh uniform keys.** BRKGA rejects settings where `int(p * pm)` is zero in every engine,
  so the mutant channel cannot be switched off by an override. See the
  [parameter reference](parameters.md).
- **No lost improvement.** The reported cost equals the lowest objective value any evaluation
  returned, and every history is non-increasing, on the Python, Rust and C++ engines
  (`test_best_cost_monotonic.py`).

The convergence statement is about the **best evaluated solution**, not about the keys. Keys keep
moving and the population need not collapse; that is not a defect.

## What it does not say

- **Nothing about speed.** The bound uses only the uniform draws. For a sort decoder over `n`
  items a single permutation has mass `1/n!`, and a symmetric TSP cycle has `2/(n-1)!`; the
  bound is correct and useless at `n = 30`. A threshold decoder with one target bit pattern has
  mass `2**-n`. Practical quality comes from the operators exploiting structure, which this page
  does not analyze.
- **`mu` is unknown for real problems.** You can compute it only for toy decoders. Do not read the
  bound as a confidence level for your instance.
- **No certificate of optimality.** A found solution is never proved optimal.
- **No finite-time quality.** Elapsed time, evaluation budgets and a hard objective all break
  nothing in the statement; they just stop the search before the guarantee matters.
- **Only the mutant channel is covered.** Other algorithms in a portfolio add search power but are
  not part of this bound. A portfolio keeps the best of everything it ran.
- **A decoder that cannot produce a solution has `mu = 0`.** Then nothing can be guaranteed, for
  example when feasibility repair always maps to the same few solutions.

## A misconfiguration the library now refuses

```python
from rko_solver import RKO

try:
    RKO(
        decode, objective, n_variables, algorithms=["BRKGA"], adaptation="fixed",
        hyperparameters={"BRKGA": {"p": 15, "pm": 0.05}}, execution="sequential",
    ).solve(0.1)
except ValueError as error:
    print(error)
```

With `p = 15` and `pm = 0.05` the mutant count rounds down to zero, so the call raises
`ValueError: BRKGA: p * pm must give at least one mutant per generation.` With the default
`pm = 0.10` the population must be at least 10.

## How this is checked

`test_brkga_survival_bound.py` uses a five-bit threshold target, `mu = 1/32`, which is the only
kind of target whose mass is exactly known.

- On the Python algorithm, an injected local step costs exactly one evaluation, so a budget of
  `p + G * (p - elite + 1)` holds exactly `G` generations. Over 2000 seeded runs per budget the
  miss count must not exceed the bound by an exact one-sided binomial test at `1e-6`.

| Generations `G` | Evaluations | Miss rate (measured) | Bound |
| --- | --- | --- | --- |
| 0 | 10 | 0.707 | 0.728 |
| 2 | 30 | 0.470 | 0.530 |
| 4 | 50 | 0.321 | 0.386 |
| 6 | 70 | 0.223 | 0.281 |
| 10 | 110 | 0.108 | 0.149 |

  Population 10, one elite, five mutants, 2000 seeds. The observed misses sit below the bound, as
  expected, because crossover also reaches the target.
- On all three engines through the public API, the test requires that every one of 60 seeds finds
  the target within 6000 evaluations. The Rust and C++ local step spends an unreported number of
  evaluations per generation, so no exact bound is claimed there; even at 30 evaluations per
  generation the guaranteed draws would put a miss below `1e-10` per seed.

Limits of that check:

- Removing the mutants, or drawing them where they cannot hit the target, makes the test fail
  (1252 and 1331 misses of 2000 against a bound of 0.53, in two temporary mutations).
- **Losing a single mutant is not detected.** Crossover compensates, and the margin between the
  measured rate and the bound is larger than the effect. The exact mutant count is protected by
  the validation tests instead.
- It is a smoke test of the library's behaviour on one toy decoder, not a proof for your problem.
