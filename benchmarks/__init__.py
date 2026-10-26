"""Reproducible standalone benchmark programs for the RKO library.

Objective:
- Collect maintainable benchmark runners and fixed reference measurements.

Technical approach:
- Keep command-line experiments outside the installed runtime package.
- Import reusable problem adapters from the public `rko_solver` package.

Flow (pseudocode):
1. Select a documented workload and build its isolated runner.
2. Execute repeatable trials and preserve raw observations.
3. Publish only explicitly identified benchmark references.

Important criteria:
- Benchmark inputs and generated binaries remain outside the repository.
- Benchmark evidence does not imply universal performance or quality.

Inputs/Outputs:
- Input: benchmark-specific CLI arguments and external workloads.
- Output: raw measurements and maintained interpretation.
"""
