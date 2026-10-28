"""Collection rules of the benchmark tests.

Objective:
- Skip the tuning harness tests in environments that do not install its optional dependencies.

Technical approach:
- Ignore `test_tuning.py` at collection time unless both `optuna` and `scipy` are importable,
  because the harness imports them at module level and the guard must run before that import.

Flow (pseudocode):
1. Look up both packages without importing them.
2. List the test module as ignored when either is missing.

Important criteria:
- The jobs that install the `tuning` group still run the tests.

Inputs/Outputs:
- Input: the installed packages.
- Output: the `collect_ignore` list pytest reads.
"""

import importlib.util

OPTIONAL = ("optuna", "scipy")
collect_ignore = (
    []
    if all(importlib.util.find_spec(name) is not None for name in OPTIONAL)
    else ["test_tuning.py"]
)
