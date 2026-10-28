"""Trusted uncommitted blending-study contracts.

Objective:
- Verify replay checks and restoration without hardware-dependent speed assertions.

Technical approach:
- Compare production with itself and with a deliberately wrong operator.

Flow (pseudocode):
1. Execute tiny paired searches through the public benchmark entry point.
2. Verify exact work, stream restoration and rejection of changed behavior.

Important criteria:
- Functional tests do not establish a performance gain.

Inputs/Outputs:
- Input: trusted candidate callables and bounded search sizes.
- Output: replay assertions or a validation error.
"""

import numpy as np
import pickle
import pytest
import random
import rko_solver.components.local_search.nelder_mead_ls as nelder_mead
from benchmarks.blending_work import BlendingStudy
from rko_solver.components.blending.blending import blending


def test_identity_candidate_preserves_work_and_restores_bindings() -> None:
    """Keep search replay, operator binding and caller random streams unchanged.

    Raises:
        AssertionError: If the study changes work, bindings or random state.
    """
    before = pickle.dumps((random.getstate(), np.random.get_state()))
    original = nelder_mead._blend
    rows = BlendingStudy.run(blending, dimension=4, evaluations=64, seeds=1)

    assert len(rows) == 6
    assert {row["regime"] for row in rows} == {"native", "python_heavy"}
    assert all(
        row["baseline"]["evaluations"] == row["candidate"]["evaluations"] == 64 for row in rows
    )
    assert all(row["baseline"]["keys_sha256"] == row["candidate"]["keys_sha256"] for row in rows)
    assert nelder_mead._blend is original
    assert pickle.dumps((random.getstate(), np.random.get_state())) == before


def test_wrong_candidate_is_rejected_and_bindings_restored() -> None:
    """Reject altered RNG consumption and offspring rather than timing a false win.

    Raises:
        AssertionError: If a changed trajectory is accepted or bindings leak.
    """
    original = nelder_mead._blend
    before = pickle.dumps((random.getstate(), np.random.get_state()))

    def wrong(a: np.ndarray, b: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """Return an invalid replacement that consumes no expected RNG draws.

        Args:
            a (np.ndarray): First parent.
            b (np.ndarray): Unused second parent.
            factor (float): Unused complement mode.

        Returns:
            np.ndarray: A detached first parent, not valid blending behavior.
        """
        return a.copy()

    with pytest.raises(ValueError, match="replay"):
        BlendingStudy.run(wrong, dimension=4, evaluations=256, seeds=1)
    assert nelder_mead._blend is original
    assert pickle.dumps((random.getstate(), np.random.get_state())) == before


def test_search_failure_restores_bindings_and_random_streams() -> None:
    """Restore state even when a valid kernel fails later inside actual search.

    Raises:
        AssertionError: If a search exception leaves a candidate or RNG installed.
    """
    before = pickle.dumps((random.getstate(), np.random.get_state()))
    original = nelder_mead._blend

    def failing(a: np.ndarray, b: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """Fail only at a search dimension outside the preliminary kernel fixtures.

        Args:
            a (np.ndarray): First parent.
            b (np.ndarray): Second parent.
            factor (float): Complement mode.

        Returns:
            np.ndarray: Unchanged production offspring outside the failure case.

        Raises:
            RuntimeError: At the chosen actual-search dimension.
        """
        if len(a) == 4:
            raise RuntimeError("candidate search failure")
        return blending(a, b, factor)

    with pytest.raises(RuntimeError, match="candidate search failure"):
        BlendingStudy.run(failing, dimension=4, evaluations=256, seeds=1)
    assert nelder_mead._blend is original
    assert pickle.dumps((random.getstate(), np.random.get_state())) == before


@pytest.mark.parametrize(
    "arguments", [{"dimension": 0}, {"evaluations": True}, {"seeds": 0}, {"seed_start": -1}]
)
def test_invalid_work_is_rejected(arguments: dict[str, object]) -> None:
    """Reject invalid diagnostic work before any execution.

    Args:
        arguments (dict[str, object]): Invalid public study options.

    Raises:
        AssertionError: If invalid work reaches the benchmark.
    """
    with pytest.raises(ValueError):
        BlendingStudy.run(blending, **arguments)
