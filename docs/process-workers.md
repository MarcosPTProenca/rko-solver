# Automatic process-worker defaults

## Behavior change

Previously, omitted/`None` `n_jobs` created one spawned search worker per selected
algorithm. It now uses the smaller of portfolio size and the process-usable logical
CPU count from `os.process_cpu_count()`, with a conservative one-worker fallback
when discovery is unavailable. This also applies to `run()` and the CLI when
`--n-jobs` is omitted. Existing positive explicit limits retain their meaning,
including limits greater than the detected CPU count. Zero, negative, boolean and
non-integer limits remain invalid. Sequential and thread execution are unchanged.

Every selected algorithm remains assigned in portfolio order, round-robin across
workers. The original common deadline remains, but the former fixed
`time_limit / ceil(number_of_algorithms / number_of_workers)` cap is replaced by
**remaining worker time / remaining worker searches**, recomputed at dispatch.
Thus groups of three and two start with approximately one third and one half of
the common budget, respectively. Early completion benefits later searches in the
same group, and overruns reduce the remaining shares. This prioritizes using each
worker's budget, not equal time for every algorithm across differently sized groups.

There is no cross-worker work stealing, search resumption/restart, or new deadline.
If the last search finishes early, remaining time cannot be reclaimed by a finished
search. Built-in searches generally run until their assigned limit; shorter groups
are the measured source of reclaimable time. Callbacks can overrun their share and
starve later searches; queued algorithms may consequently report `not_started`.
Their `SearchMetadata.time_limit` is zero because no search allocation was granted;
started reports retain the actual dispatch-time allocation, excluding queueing.
These are maxima, not consumed time: after an early return, the sum of reported
limits may exceed the worker's total budget even though its actual elapsed searches
remain within the common deadline. Use `elapsed_seconds` to account for time spent.

**Changing worker counts or the allocation policy changes timed trajectories,
cooperation and per-algorithm budgets.** This is not a promise of equal-work speedups or better solution quality.
Sequential algorithms in a worker share callback instances without resetting
mutable state. IPR now seeds a missing elite pair within its own allocation, so it
need not wait for a producer queued behind it. A pool must still retain at least
two entries for actual relinking; initialization does not repair domain constraints. Owned library RNG seeds remain assigned by portfolio position, not by
worker count, but timed/cooperative final results need not match.

## Preserve dedicated workers

Choose a portfolio explicitly and request its size:

```python
algorithms = ["SA", "GA", "PSO"]
optimizer = RKO(decoder, objective, dimension, algorithms=algorithms,
                n_jobs=len(algorithms))
```

For the current shipped ten-algorithm CLI portfolio, use `--n-jobs 10`.
A one-search worker still receives the time left until the common deadline.
No legacy fixed-slice switch, `-1`/`all` sentinel or new CLI flag is introduced.
Sequential and thread scheduling are unchanged.

## Scheduling measurements

The [remaining-time reference](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md#remaining-time-worker-scheduling)
compares the previous policy with the new one under the same search budget and
worker counts. For ten searches/four workers, median unused worker-session time
fell from about 0.80 seconds to 0.002–0.006 seconds out of 4.8 available worker-seconds.
This is search-session wall-time accounting, **not CPU utilization or a general
speed/quality improvement**. Evaluation counts and costs were mixed across neutral
problems; do not reinterpret prior worker-count references as equal-work gains.
Formerly shorter groups now stay alive longer within the allowed budget; this is
not a measured reduction in CPU consumption, energy or resident-memory lifetime.

## Resource accounting

`result.n_jobs` retains the requested limit (`None` for automatic). New
`result.available_cpus` records the usable logical CPU discovery once per process
solve; it is `None` for nonprocess runs and legacy manually constructed results.
The existing `result.n_workers` property uses that immutable snapshot and the
requested cap. Later affinity changes do not rewrite earlier results, and the
next solve discovers capacity again. Legacy six-argument `RKOResult` construction
keeps its historical worker estimate when no capacity snapshot exists.

The cap covers **search processes only**, not the pool manager, coordinator,
resource tracker, callback subprocesses or BLAS/OpenMP/native threads. It is per
solve, not a machine-wide allocator shared by concurrent RKO instances. Affinity
and Python CPU-count overrides are respected, but not every container CPU quota
is reflected in the discovered count; supply a positive explicit `n_jobs` for such
limits. The library does not change environment variables or native thread settings
inside an already running application.

See [the resource benchmark](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md#bounded-process-worker-resources)
for measured scope and limitations. Installation/filesystem startup gains remain a
separate [WSL setup concern](running.md#wsl-startup-and-virtual-environment-placement).
