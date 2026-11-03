"""Verify categorical intervals and their finite-precision right inverse.

Objective:
- Protect legal action support and deterministic interval-boundary behavior.

Technical approach:
- Compare distributions with independently calculated categorical probabilities.

Flow (pseudocode):
1. Construct intervals from scores and a positive uniform probability mixture.
2. Select actions and encode them as interior midpoints.
3. Check known probabilities and interval ownership.

Important criteria:
- Tests use the public numerical interface rather than internal normalization steps.

Inputs/Outputs:
- Input: finite action scores and valid random keys.
- Output: assertions on probabilities and action selection.
"""

import numpy as np
import pytest
from benchmarks.quantile.probabilities import Intervals
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st


def test_mixed_probabilities_have_known_mass() -> None:
    """A one-to-three score ratio with a one-fifth mixture gives masses 0.3 and 0.7."""
    intervals = Intervals.from_logits(np.log([1.0, 3.0]), mixture=0.2)
    np.testing.assert_allclose(intervals.probabilities, [0.3, 0.7], atol=1e-15)


def test_half_open_boundaries_and_midpoint_inverse() -> None:
    """A boundary belongs to its following action and both midpoints replay their action."""
    intervals = Intervals.from_logits(np.log([1.0, 3.0]), mixture=0.2)
    edge = intervals.bounds[1]
    assert intervals.select(0.0) == 0
    assert intervals.select(np.nextafter(edge, 0.0)) == 0
    assert intervals.select(edge) == 1
    assert intervals.select(np.nextafter(1.0, 0.0)) == 1
    for action in range(2):
        assert intervals.select(intervals.midpoint(action)) == action


@pytest.mark.parametrize("key", [-0.1, 1.0, float("nan"), float("inf")])
def test_selection_rejects_invalid_keys(key: float) -> None:
    """Reject keys outside the finite half-open domain.

    Args:
        key (float): Invalid decision key.
    """
    with pytest.raises(ValueError, match="key"):
        Intervals.from_logits(np.zeros(2)).select(key)


@pytest.mark.parametrize("scores", [[], [[1.0, 2.0]], [float("nan")], [float("inf")]])
def test_invalid_action_scores_are_rejected(scores: list[object]) -> None:
    """Reject an empty, multidimensional or nonfinite action-score vector.

    Args:
        scores (list[object]): Invalid input scores.
    """
    with pytest.raises(ValueError, match="scores"):
        Intervals.from_logits(np.asarray(scores))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_temperature_is_rejected(temperature: float) -> None:
    """Require a finite positive temperature.

    Args:
        temperature (float): Invalid temperature.
    """
    with pytest.raises(ValueError, match="temperature"):
        Intervals.from_logits(np.zeros(2), temperature=temperature)


@pytest.mark.parametrize("mixture", [0.0, -1.0, 1.0, float("nan")])
def test_invalid_uniform_mixture_is_rejected(mixture: float) -> None:
    """Require a strict positive probability mixture below one.

    Args:
        mixture (float): Invalid uniform mixture coefficient.
    """
    with pytest.raises(ValueError, match="mixture"):
        Intervals.from_logits(np.zeros(2), mixture=mixture)


def test_collapsed_support_is_an_explicit_error() -> None:
    """Do not silently accept a probability floor that underflows to zero."""
    with pytest.raises(ValueError, match="collapsed"):
        Intervals.from_logits(np.array([0.0, -1000.0]), mixture=np.nextafter(0.0, 1.0))


def test_midpoint_requires_a_representable_interior() -> None:
    """An adjacent-float interval has no representable strict midpoint."""
    intervals = Intervals(np.array([1.0, 1e-16]), np.array([0.0, np.nextafter(1.0, 0.0), 1.0]))
    with pytest.raises(ValueError, match="interior"):
        intervals.midpoint(1)


@settings(max_examples=80, derandomize=True, deadline=None)
@given(
    st.lists(st.floats(min_value=-1e308, max_value=1e308), min_size=1, max_size=20),
    st.floats(min_value=0.0, max_value=np.nextafter(1.0, 0.0)),
)
def test_every_action_keeps_support_and_reencodes(scores: list[float], key: float) -> None:
    """Extreme finite scores retain legal support and an exact selected-action inverse.

    Args:
        scores (list[float]): Generated finite action scores.
        key (float): Generated valid decision key.
    """
    intervals = Intervals.from_logits(np.asarray(scores), temperature=np.nextafter(0.0, 1.0))
    assert np.all(np.diff(intervals.bounds) > 0.0)
    action = intervals.select(key)
    assert 0 <= action < len(scores)
    assert intervals.select(intervals.midpoint(action)) == action
    for expected in range(len(scores)):
        assert intervals.select(intervals.midpoint(expected)) == expected
