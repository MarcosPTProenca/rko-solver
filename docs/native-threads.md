# Native threads in numerical callbacks

`n_jobs` limits RKO search processes, **not BLAS/OpenMP threads**. A four-worker
portfolio with four OpenBLAS threads per worker can request sixteen BLAS threads
on four usable CPUs. Configured team sizes are not proof that every thread is
active; idle native pools alone do not demonstrate CPU oversubscription.

## Configure the application launch, before importing NumPy

For a process-parallel numerical workload, one native thread per search worker is
a useful starting point to measure, not a universal optimum. In Bash/WSL, prefix
your normal command with the relevant backend settings. For example, this
fully runnable inspection command starts NumPy with one-thread settings:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  uv run --locked --with threadpoolctl python -m threadpoolctl -i numpy
```

Replace the diagnostic command after the environment prefix with your existing
`uv run --locked python your_solver.py` or `uv run --locked python -m examples.run_kp_cpp_instances ...` invocation.
Those are templates for your own script/arguments, not bundled filenames. The
prefix applies only to that command and its descendants; it does not persistently
export settings to the shell or change the RKO default. `threadpoolctl` is needed
only for inspection/the benchmark, **not for normal solves**.

For Python applications or Windows shells, launch a fresh interpreter with an
explicit environment rather than changing a live application's native pools.
This template preserves the caller's environment; replace `your_solver_module`
with your application's guarded module:

<!-- doc-test: skip launcher template for the reader's own your_solver_module -->
```python
import os
import subprocess
import sys

if __name__ == "__main__":
    settings = dict.fromkeys(
        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"),
        "1",
    )
    subprocess.run(
        [sys.executable, "-m", "your_solver_module"],
        env=os.environ | settings,
        check=True,
    )
```

This approach also applies when the caller has already imported NumPy: the new
interpreter imports its own libraries using the supplied environment, and spawned
RKO children inherit it. Changing `os.environ` **after** importing a native library
is not a reliable way to reconfigure the library already loaded in that process.
Do not put unguarded process creation in the solver module.

OpenBLAS pthread builds prioritize `OPENBLAS_NUM_THREADS`; OpenMP builds use
`OMP_NUM_THREADS` instead. MKL/BLIS controls are backend-specific; domain-specific
settings or other libraries can require additional configuration. Inspect the
actual runtime and verify your workload. These settings do not limit arbitrary
callback thread pools, GPU runtimes, other applications or concurrent RKO solves.
Keep `n_jobs` within the resources allocated to this application as well.

## Why RKO does not silently change native pools

Wrapping only the parent `solve()` in `threadpoolctl.threadpool_limits(...)` does
**not** configure fresh spawned workers. Our negative control reports one BLAS
thread in the parent but four in each worker under the inherited environment.
Runtime controllers act on libraries loaded in their process; libraries imported
later also need separate consideration. With the measured OpenBLAS pthread build,
changing a runtime limit affects the entire current process, potentially disturbing
other numerical work or simultaneous thread-backend solves.

Consequently this change adds no native-thread keyword, runtime dependency,
parent-environment mutation, private vendor `ctypes` calls or implicit global
policy to RKO. Launch-time controls are explicit and inherited without those
cross-solve races. A future automatic controller needs a separately supported
scope/vendor contract; it is not implemented here.

## Reproduce and verify workers

```bash
uv run --locked --with threadpoolctl==3.7.0 python -m benchmarks.runtime.native_threads
```

The default benchmark alternates inherited/one-thread launch environments over
three repeats, three neutral numerical objectives and one/four dedicated workers.
It verifies actual **worker** native-team counts, assigned initial seeds, identical
candidate-stream hashes, finite completed evaluation counts and independent result
costs. Parent environment/runtime limits are checked unchanged. Existing inherited
caps are honored, not cleared to manufacture a baseline. On a machine already
limited to one native thread, the two launch policies may be identical.

Populations are deliberately larger than the 64-evaluation prefix: no full
generation or asynchronous elite feedback is measured. Dense objectives use
512-dimensional keys and fixed matrices. Warm-up is excluded from callback timing
but included in search/end-to-end timing. The private stopping signal in this
benchmark is instrumentation, **not a supported user callback interface**.

The [identified Linux reference](https://github.com/MarcosPTProenca/rko-solver/blob/main/benchmarks/references/native-threads-reference.jsonl)
records OpenBLAS 0.3.31.dev, NumPy 2.4.3, Python 3.14.8 and four usable CPUs.
In its four-worker dense-matrix prefix, inherited/one-thread medians were:

| Metric | Inherited (4 BLAS threads/worker) | One thread/worker |
| --- | ---: | ---: |
| Slowest worker search prefix | 6.016 s | 3.491 s |
| End-to-end solve | 7.564 s | 4.860 s |
| Sum of measured callback process CPU | 18.574 s | 7.967 s |

A separate full repetition also improved this dense workload, but absolute timings
varied. Single-worker dense callbacks also improved at this matrix size, so the
measurements do not isolate outer-process oversubscription as the sole cause:
per-kernel team overhead also matters. Cheap/vector objectives showed mixed search
timings. These single-worker results are not evidence that single-threaded BLAS
always wins. These
are fixed initialization-prefix measurements, **not full-portfolio speedups,
solution-quality improvements or C++/Rust parity**. BLAS summation order can change
floating-point costs, so numerical checks use tolerance. Windows CI runs only a
tiny correctness/configuration smoke case, not this performance experiment.

Sources:
- [NumPy 2.4 global performance configuration](https://numpy.org/doc/2.4/reference/global_state.html).
- [NumPy runtime inspection](https://numpy.org/doc/2.4/reference/generated/numpy.show_runtime.html).
- [OpenBLAS threading controls and precedence](https://github.com/OpenMathLib/OpenBLAS#setting-the-number-of-threads-using-environment-variables).
- [threadpoolctl scope and limitations](https://github.com/joblib/threadpoolctl#known-limitations).
