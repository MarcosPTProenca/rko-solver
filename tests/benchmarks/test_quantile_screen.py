"""Shared-checkpoint arm construction checks for the F5 development screen.

Objective:
- Verify that every arm receives the same checked start and the same wall-time budget.

Technical approach:
- Run a real tiny RKO solver per arm and check completed search metadata, checked
  initial keys and the classical arm's scorer-free codec.

Flow (pseudocode):
1. Build a tiny tour task with a tiny frozen scorer.
2. Build every arm's solver and check its declared algorithms and allowance.
3. Solve each arm and check the result reports the common time allowance.

Important criteria:
- The classical arm never receives the frozen scorer in its codec.
- The portfolio arm's three members each report the same `time_limit`.

Inputs/Outputs:
- Input: tiny deterministic task and scorer fixtures.
- Output: assertions on arm configuration and completed search results.
"""

import numpy as np
import pytest
from benchmarks.quantile import screen
from benchmarks.quantile.model import FrozenScorer
from benchmarks.quantile.tasks import Tour


def _scorer(family: str) -> FrozenScorer:
    """Build a tiny deterministic frozen scorer for one feature family.

    Args:
        family (str): 'tour' or 'cover'.

    Returns:
        FrozenScorer: Tiny two-hidden-unit scorer.
    """
    rng = np.random.default_rng(0)
    return FrozenScorer(
        family, rng.normal(size=(4, 2)), np.zeros(2), rng.normal(size=2), np.array(0.0)
    )


def test_builder_rejects_an_unknown_arm() -> None:
    """An unrecognized arm name is rejected before any solver is built."""
    task = Tour(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    builder = screen.ArmBuilder(task, _scorer("tour"))
    with pytest.raises(ValueError, match="arm"):
        builder.bridge("adversarial")


def test_builder_rejects_a_scorer_from_the_wrong_family() -> None:
    """A cover-family scorer cannot drive a tour task's arms."""
    task = Tour(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="family"):
        screen.ArmBuilder(task, _scorer("cover"))


def test_classical_arm_never_receives_the_frozen_scorer() -> None:
    """Classical construction stays scorer-free; every other arm uses the frozen scorer."""
    task = Tour(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    builder = screen.ArmBuilder(task, _scorer("tour"))
    assert builder.bridge("classical").codec.scorer is None
    for arm in ("independent", "genetic", "portfolio"):
        assert builder.bridge(arm).codec.scorer is not None


def test_every_arm_starts_from_the_same_reference_trace() -> None:
    """Every arm's checked starting key reconstructs the identical shared reference trace."""
    task = Tour(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    builder = screen.ArmBuilder(task, _scorer("tour"))
    reference = builder.reference_trace()
    for arm in screen.ARMS:
        codec = builder.bridge(arm).codec
        keys = np.array(builder.initial_key(arm, reference))
        assert codec.decode(keys).trace == reference


@pytest.mark.parametrize("arm", screen.ARMS)
def test_every_arm_completes_under_the_common_wall_time_allowance(arm: str) -> None:
    """Each arm's real RKO result respects the shared allowance without multiplying it.

    Single-algorithm arms report the full allowance; the three-member portfolio
    arm divides that same allowance across its members under `n_jobs=1`, exactly
    as any single-worker multi-algorithm queue divides remaining time.

    Args:
        arm (str): One of `screen.ARMS`.
    """
    task = Tour(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    builder = screen.ArmBuilder(task, _scorer("tour"))
    reference = builder.reference_trace()
    result = builder.run(arm, seed=7, time_limit=2.0, reference=reference)
    assert np.isfinite(result.cost)
    assert result.algorithms == screen.ARM_ALGORITHMS[arm]
    total_assigned = sum(search.time_limit for search in result.searches)
    # Divided, never multiplied: process startup may take a little of the 2.0 s, and the
    # three-member portfolio would sum to about 6.0 if each member kept the whole allowance.
    assert 1.0 < total_assigned <= 2.01
