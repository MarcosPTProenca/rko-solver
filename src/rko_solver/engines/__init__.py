"""Python-side adapters of the engines that run the search.

Objective:
- Group the code that connects the public `rko_solver` API to each engine implementation.

Technical approach:
- The Python engine itself lives in `rko_solver.metaheuristics`, `rko_solver.components` and
  `rko_solver.orchestration`, because those modules are the public algorithm API.
- `rko_solver.engines.rust` holds the adapters of the compiled engines (Rust and C++), which both
  run through the `rko_solver._native` extension.

Flow (pseudocode):
1. `RKO` validates `engine` and `execution`.
2. The python engine runs in-process or in worker processes.
3. The rust and cpp engines run through `rko_solver.engines.rust.runner.NativeRunner`.

Important criteria:
- This package imports nothing at import time, so `import rko_solver` never loads the extension.

Inputs/Outputs:
- Input: none.
- Output: subpackages named after the engine families.
"""
