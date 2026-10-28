"""Example entrypoints for running RKO on concrete benchmark instances.

Objective:
- Group executable examples that demonstrate end-to-end use of the RKO API.

Technical approach:
- Provide modules that read instance data, define decoder/cost callbacks,
  and invoke the public `rko_main.run` orchestration entrypoint.

Flow (pseudocode):
1. Parse problem data from external files.
2. Build decoder and objective callbacks.
3. Execute RKO and report best solution metrics.

Important criteria:
- Keep examples deterministic in structure and easy to adapt.

Inputs/Outputs:
- Input: benchmark files and command-line options.
- Output: printed run summaries and decoded best solutions.
"""
