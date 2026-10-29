"""Repository-only bounded model-conversion validation, provenance and overhead studies.

Objective:
- Keep purpose-specific benchmark code outside the installed runtime package.

Technical approach:
- Keep study modules and reusable helpers in one purpose-specific namespace.
- Expose guarded module entry points without running a study on package import.

Flow (pseudocode):
1. Import a named study or helper from this package.
2. Select explicit workloads and resource budgets through its entry point.
3. Retain source-identified results with the study's measurement limits.

Important criteria:
- Importing this package does not execute a benchmark or alter solver behavior.

Inputs/Outputs:
- Input: explicit study parameters or reusable helper arguments.
- Output: benchmark observations or helper results from the selected module.
"""
