"""Deterministic particle movement equivalence checks.

Objective:
- Preserve PSO velocity, position, RNG and boundary-reversion contracts.

Technical approach:
- Compare production movement against the former scalar formula with owned RNGs.
- Exercise dimensions on both sides of the NumPy batching threshold.

Flow (pseudocode):
1. Construct matching positions, velocities and detached personal/global bests.
2. Replay identical Python random streams through scalar and production movement.
3. Assert exact arrays, RNG states, unchanged bests and inclusive/exclusive bounds.

Important criteria:
- Genes are independent; particles must still move sequentially in solve().
- Exact zero is valid, exact one is reverted with its velocity set to zero.

Inputs/Outputs:
- Input: deterministic key vectors, velocities and acceleration coefficients.
- Output: exact replay and in-place ownership assertions.
"""

import numpy as np
import pytest
import random
from rko_solver.metaheuristics.pso import PSO


@pytest.mark.parametrize("dimension", [1, 8, 31, 32, 33, 100, 1000])
@pytest.mark.parametrize("coefficients", [(0.0, 0.0, 0.0), (2.0, 2.0, 0.7), (1.5, 2.5, 0.9)])
def test_movement_matches_scalar_formula(
    dimension: int, coefficients: tuple[float, float, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Match exact motion and RNG state without changing personal/global best storage.

    Args:
        dimension (int): Gene count around and above the batching threshold.
        coefficients (tuple[float, float, float]): Cognitive, social and inertia values.
        monkeypatch (pytest.MonkeyPatch): Substitute an independent Python RNG.
    """
    data_rng = np.random.default_rng(17)
    position = data_rng.random(dimension)
    velocity = data_rng.uniform(-2, 2, dimension)
    personal_best = data_rng.random(dimension)
    global_best = data_rng.random(dimension)
    expected_position = position.copy()
    expected_velocity = velocity.copy()
    saved_personal, saved_global = personal_best.copy(), global_best.copy()
    reference_rng, actual_rng = random.Random(42), random.Random(42)
    c1, c2, inertia = coefficients
    for j in range(dimension):
        r1, r2 = reference_rng.random(), reference_rng.random()
        expected_velocity[j] = inertia * (
            expected_velocity[j]
            + c1 * r1 * (personal_best[j] - expected_position[j])
            + c2 * r2 * (global_best[j] - expected_position[j])
        )
        old_key = expected_position[j]
        expected_position[j] = expected_position[j] + expected_velocity[j]
        if expected_position[j] < 0.0 or expected_position[j] >= 1.0:
            expected_position[j] = old_key
            expected_velocity[j] = 0.0
    monkeypatch.setattr(random, "random", actual_rng.random)
    PSO._move_particle(position, velocity, personal_best, global_best, c1, c2, inertia)
    np.testing.assert_array_equal(position.view(np.uint64), expected_position.view(np.uint64))
    np.testing.assert_array_equal(velocity.view(np.uint64), expected_velocity.view(np.uint64))
    np.testing.assert_array_equal(personal_best, saved_personal)
    np.testing.assert_array_equal(global_best, saved_global)
    assert actual_rng.getstate() == reference_rng.getstate()


@pytest.mark.parametrize("dimension", [4, 32])
def test_boundary_reversion_is_per_gene(dimension: int) -> None:
    """Revert invalid genes, not the complete particle, while accepting exact zero.

    Args:
        dimension (int): Scalar or batched gene count.
    """
    position = np.full(dimension, 0.5)
    velocity = np.resize([0.5, -0.5, -0.6, 0.1], dimension)
    best = position.copy()
    saved_state = random.getstate()
    try:
        PSO._move_particle(position, velocity, best, best, 0.0, 0.0, 1.0)
    finally:
        random.setstate(saved_state)
    np.testing.assert_array_equal(position, np.resize([0.5, 0.0, 0.5, 0.6], dimension))
    np.testing.assert_array_equal(velocity, np.resize([0.0, -0.5, 0.0, 0.1], dimension))
