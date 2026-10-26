"""Opt-in research interfaces, separate from the stable optimizer API.

Objective:
- Keep experimental application capabilities out of core RKO imports.
Technical approach:
- Expose subpackages explicitly without importing their execution controllers.
Flow (pseudocode):
1. Import the desired experimental subpackage explicitly.
2. Follow its qualification and trust contract before execution.
Important criteria:
- No stability or production-readiness guarantee is implied.
Inputs/Outputs:
- Input: explicit experimental imports.
- Output: opt-in research capabilities only.
"""
