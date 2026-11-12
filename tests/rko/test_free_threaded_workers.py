"""Contract tests for the default worker count of Python callbacks on the Rust engine.

Objective:
- Verify that Python callbacks keep one native worker while the interpreter lock is enabled, and
  get no forced limit when the lock is disabled (free-threaded CPython).

Technical approach:
- Test the pure decision function directly, so no native extension is needed to check the rule.

Flow (pseudocode):
1. Ask for the default worker limit with the lock enabled and expect one worker.
2. Ask with the lock disabled and expect no forced limit.

Important criteria:
- An explicit `n_jobs` always wins; this decision only supplies the default.

Inputs/Outputs:
- Input: whether the interpreter lock is enabled.
- Output: a worker limit, or None when no limit is forced.
"""

import pytest
from rko_solver.engines.rust.runner import default_callback_workers


def test_callbacks_keep_one_worker_while_the_interpreter_lock_is_enabled() -> None:
    """The lock serializes Python callbacks, so one native worker is the honest default."""
    assert default_callback_workers(gil_enabled=True) == 1


def test_callbacks_get_no_forced_limit_without_the_interpreter_lock() -> None:
    """Without the lock, callbacks can run on several native workers at once."""
    assert default_callback_workers(gil_enabled=False) is None


@pytest.mark.parametrize("gil_enabled", [True, False])
def test_decision_returns_only_a_default_not_a_fixed_count(gil_enabled: bool) -> None:
    """The result is either None or a positive count; it never silently exceeds one.

    Args:
        gil_enabled (bool): Whether the interpreter lock is enabled.
    """
    value = default_callback_workers(gil_enabled=gil_enabled)
    assert value is None or value == 1
