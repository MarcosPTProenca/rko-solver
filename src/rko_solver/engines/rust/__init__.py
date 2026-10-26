"""Python adapters of the compiled Rust and C++ engines.

Objective:
- Keep everything the compiled engines need on the Python side in one place.

Technical approach:
- `runner` snapshots validated parameters and calls the `rko_solver._native` extension.
- `batch` adapts vectorized block objectives and `compiled` prepares Numba objectives.

Flow (pseudocode):
1. `rko_solver.problem` builds a problem definition using `batch` or `compiled`.
2. `RKO` passes the problem and validated settings to `runner.NativeRunner`.
3. The runner returns keys, cost and per-search metadata.

Important criteria:
- Submodules are imported on demand; this package imports none of them.

Inputs/Outputs:
- Input: none.
- Output: the `batch`, `compiled` and `runner` submodules.
"""
