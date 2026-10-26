# Run RKO on an instance

## Terminal command

After `uv sync --locked --group dev`, run from the repository root:

```bash
uv run --locked rko-solver src/rko_solver/examples/instances/kp50.txt \
  --limit-time 10 --execution process --random-state 42 \
  --output-csv results/kp50.csv
```

This is the Knapsack adapter shipped with the library, analogous to the supplied
C++ `Problem.h`. It reads the instance, constructs `RKO`, runs the configured
portfolio, prints cost/profit/weight/feasibility and exports CSV. Use quotes around
paths containing spaces. In PowerShell put the command on one line or replace
Bash continuation backslashes with PowerShell backticks.

`uv run --locked rko-solver --help` lists options. The equivalent installed command is
`rko-solver ...`; running the Python module directly also remains supported. Startup
is additional to the process search budget; see README for stopping limits.
The packaged default selects ten algorithms. Without `--n-jobs`, process mode
uses at most the smaller of portfolio size and `os.process_cpu_count()` usable
logical CPUs (one if discovery is unavailable). `--n-jobs 4` explicitly requests
up to four search workers plus the pool manager, even if fewer CPUs are available. Assignments are round-robin; each worker executes
its group sequentially. At each dispatch its remaining common-deadline time is
shared equally among that worker's remaining searches. Ten algorithms on four
workers form groups of 3, 3, 2 and 2: initial shares are approximately one third
or one half of the budget, rather than the previous fixed one-third cap for all.
Early returns leave more time for the same worker's queued searches; no search
receives a new portfolio deadline. Startup remains additional. There is no
cross-worker stealing or restart of completed searches; if a worker's final search
finishes early, its remaining time is still unused.
Callback overruns can consume later searches' slices. This changes timed search
trajectories and cooperation, not the configured algorithm selection or seed
assignment. Native-library threads are not capped by `--n-jobs`, and concurrent
RKO instances do not share a global worker budget. CPU discovery is affinity-aware,
not a guarantee of honoring every container quota; choose an explicit budget when
needed. Sequential and thread modes are unchanged. For the historical dedicated-worker
behavior with the shipped ten algorithms, use `--n-jobs 10`.
See [migration and resource limits](process-workers.md). For BLAS-heavy callbacks,
[native-thread launch settings](native-threads.md) configure fresh interpreters and
their spawned workers without mutating the parent application's native pools.

## WSL startup and virtual-environment placement

On WSL, putting `.venv` under `/mnt/c/...` also puts NumPy and many Python imports
on the Windows-mounted filesystem. Every spawned manager/worker reimports its
entry module. For short solves, this can dominate actual search work.

Keep the existing checkout and environment intact while testing an additional
Linux-local environment. From the repository root in a WSL shell:

```bash
RKO_ENV="$HOME/.cache/rko-python-venv"
UV_PROJECT_ENVIRONMENT="$RKO_ENV" uv sync --locked
UV_PROJECT_ENVIRONMENT="$RKO_ENV" uv run --locked python benchmarks/startup_stages.py
```

This remains an **editable development installation**: dependencies are local to
Linux, but Python still reads RKO source and parameter files from the checkout.
Keep using the environment prefix on subsequent commands; plain `uv run` selects
the normal project environment again. No global setting or existing `.venv` needs
to be removed. Use a dedicated path for this project, not a shared environment
for unrelated repositories.

For installed-package measurements, also place the installed RKO files on Linux:

```bash
UV_PROJECT_ENVIRONMENT="$RKO_ENV" uv sync --locked --no-editable
UV_PROJECT_ENVIRONMENT="$RKO_ENV" uv run --locked --no-editable \
  python benchmarks/startup_stages.py
```

A non-editable installation is a built package, not a live source view. Do not
assume source edits are visible there: rebuild/reinstall for a new artifact, or
return to editable development with `uv sync --locked` using the same environment
prefix. Retain `--no-editable` on installed-package runs so project synchronization
does not silently switch the installation mode back.

The [identified 18-solve WSL reference](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/README.md#spawn-startup-and-installation-layout)
reduced end-to-end medians from 5.51–6.17 seconds to 1.56–1.79 seconds by changing
installation layout while preserving source bytes and locked versions. That is
not a code speedup, a portable latency guarantee or steady-state throughput/quality
evidence. Startup remains additional to `time_limit`, and callback imports/data
can add their own cost. Native Windows behavior was not measured by this comparison.

## Live progress

The same short command now shows progress automatically:

```bash
uv run rko-solver src/rko_solver/examples/instances/kp50.txt
```

It prints the engine/portfolio at startup, compact new-best events immediately,
and a pool status approximately once per second, even without improvements.
For example (illustrative values and timing):

```text
[2026-10-18 15:30:01 +2.103s] NEW BEST algorithm=SA cost=-140 best=-140
[2026-10-18 15:30:02 +3.001s] STATUS best=-140 pool=1/10 [-140 (SA)]
[2026-10-18 15:30:03 +4.001s] STATUS best=-140 pool=1/10 [-140 (SA)]
```

Timestamps show local wall-clock time and monotonic elapsed solve seconds,
including configuration/startup. `NEW BEST` identifies the producing algorithm
and new shared-pool minimum. `STATUS` shows occupancy/capacity and sorted elite
costs with their sources. An initial empty snapshot, and a final snapshot before
completion, are also emitted. Chromosomes are not dumped, and full pool contents
are not repeated for every insertion.

Search improvements can genuinely cluster within a fraction of a second; records
are never delayed or replayed to simulate steady improvement. Periodic status
keeps the terminal active during plateaus. It indicates an unfinished portfolio,
not proof that callbacks are unblocked or objectives are improving. Rejected and
unshared local evaluations are not logged. A successful run ends with `RKO DONE`;
a best evaluation recovered at a budget boundary may not yet have entered the
shared pool. Failed runs propagate errors and stop the reporter without a false
completion message.

Use `--quiet` for the previous final-result-only display. Library calls remain
silent by default; request the same progress with `RKO(..., verbose=True)` or
legacy `run(..., verbose=True)`. Process output comes from the authoritative
pool manager; thread output is protected by the pool lock. The status thread is
explicitly stopped and joined when the solve exits, including failures. Lines are
flushed immediately, including when stdout is redirected. Output/blocking system
scheduling can delay status and affect timed performance; keep library benchmarks
quiet.

## Your configuration directory

`--params-dir` selects a directory, not a single file. It must contain
`config.toml` and the selected algorithms' parameter TOMLs. To copy the packaged
parameters without depending on source paths, run this once:

```bash
uv run --locked python -c "import shutil; from rko_solver.parameters.loader import default_params_dir; shutil.copytree(default_params_dir(), 'configs/my-run', ignore=shutil.ignore_patterns('*.py', '__pycache__'))"
```

This refuses to overwrite an existing directory. Edit
`configs/my-run/config.toml`, for example:

```toml
[run]
control = 1
pool_size = 10

[active_metaheuristics]
list = ["SA", "BRKGA", "GA", "PSO"]
```

Then run:

```bash
uv run --locked rko-solver path/to/instance.txt \
  --params-dir configs/my-run --limit-time 60 \
  --execution process --n-jobs 4 --random-state 42 \
  --output-csv results/my-instance.csv
```

Each algorithm's `.toml` contains `[offline]` scalar values and `[online]` named
value grids. `control = 0` selects offline settings; `control = 1` enables online
control for algorithms that implement it. SA's reference adaptive path remains
[open compatibility work](cpp-compatibility.md).

Do not edit installed package files. Do not copy C++ `config_tests.conf`,
`ParametersOffline.txt` or `ParametersOnline.txt` directly: their formats are not
TOML. C++ `strategy`, `restart`, `MAXRUNS` and `debug` are not currently supported
Python run settings; unsupported keys are rejected, not silently honored.
Repeat the command for separate trials. Current constraints are in
[status.md](status.md).

## Instance format

The current terminal adapter expects integer Knapsack data: item count and
capacity, followed by one profit/weight pair per item, separated by whitespace.
For example, an instance with three items and capacity five:

```text
3 5
6 1
10 2
12 3
```

There is no implicit parsing of arbitrary optimization instances. For a different
problem, provide its reader, random-key decoder and objective through the Python
`RKO` class, corresponding to adapting `Problem.h` in C++. A generic problem-plugin
terminal interface is not implemented by this command.

## Existing batch selection

Omit the positional instance to run a directory/glob selection:

```bash
uv run --locked rko-solver --instances-dir path/to/instances --pattern "kp*.txt" \
  --params-dir configs/my-run --limit-time 60 --n-jobs 4 \
  --output-csv results/batch.csv
```

A single positional instance cannot be combined with batch selectors. Without
any selectors, the bundled `kp10.txt` and `kp50.txt` references are used. Results
are written below the caller's `results/` directory unless `--output-csv` selects
a different path. Generated result directories are ignored by Git.
