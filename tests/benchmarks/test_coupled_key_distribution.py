"""Round-trip and full-dimensionality tests for the conditional key generator.

Objective:
- Detect loss of search directions before another timed experiment.

Technical approach:
- Fit on deterministic synthetic archives, invert generated keys and measure a numerical Jacobian.

Flow (pseudocode):
1. Fit the conditional generator.
2. Transform interior latent vectors and invert them.
3. Check that all coordinate directions survive and the fitted model can be serialized.

Important criteria:
- The Jacobian must have full rank, unlike a low-dimensional bottleneck generator.
- Timing cutoffs must leave a usable non-neural fallback.

Inputs/Outputs:
- Input: elite archives and interior random keys.
- Output: assertions on round trips, rank and fallback behavior.
"""

import numpy as np
import pickle
from benchmarks.engines.coupled_key_distribution import CoupledDistribution
from tests.benchmarks.test_neural_key_adapter import archive


def test_conditional_generator_round_trips_interior_keys() -> None:
    """Every interior latent vector has an equivalent inverse representation."""
    model = CoupledDistribution.fit(archive(3), 7)
    assert model.weights is not None and model.error < 0.9
    for keys in np.random.default_rng(5).uniform(0.1, 0.9, (20, 12)):
        transformed = model.transform(keys)
        np.testing.assert_allclose(model.inverse(transformed), keys, atol=1e-9)
        assert np.all(transformed > 0.0) and np.all(transformed < 1.0)
    keys = np.full(12, 0.5)
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(model)).transform(keys), model.transform(keys)
    )


def test_neural_generator_preserves_all_coordinate_directions() -> None:
    """A numerical Jacobian retains rank equal to the original key dimension."""
    model = CoupledDistribution.fit(archive(2), 3)
    keys = np.full(12, 0.5)
    step = 1e-5
    jacobian = np.column_stack(
        [
            (model.transform(keys + step * row) - model.transform(keys - step * row)) / (2 * step)
            for row in np.eye(12)
        ]
    )
    assert np.linalg.matrix_rank(jacobian) == 12


def test_deadline_fallback_remains_invertible() -> None:
    """Skipping training leaves the marginal full-dimensional map rather than a random network."""
    model = CoupledDistribution.fit(archive(2), 3, deadline=0.0)
    assert model.epochs == 0 and model.weights is None
    keys = np.linspace(0.1, 0.9, 12)
    np.testing.assert_allclose(model.inverse(model.transform(keys)), keys, atol=1e-9)
