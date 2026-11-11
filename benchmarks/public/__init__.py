"""Public benchmark instances with verified inputs and reference values (issue 156).

Objective:
- Keep public-instance measurements separate from the installed solver.

Technical approach:
- Separate verified sources, parsers, objectives, study orchestration and reports.

Flow (pseudocode):
1. Download and hash-check public instances.
2. Run identical decoders on the selected engines.
3. Independently check solutions and summarize the recorded evidence.

Important criteria:
- No problem-specific decoder enters the solver package.

Inputs/Outputs:
- Input: Public instances and experiment configuration.
- Output: Verified measurements and descriptive reports.
"""
