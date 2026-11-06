"""Production-specification export contracts without optimizer measurements.

Objective:
- Catch stale hardcoded parameters before a repository-only utility study runs.

Technical approach:
- Compare exports with the real native-runner factory and known packaged values.
- Require exact numerical round trips and reject mismatched native echoes.

Flow (pseudocode):
1. Load production offline specifications for several dimensions.
2. Compare parameter ordering and deterministic fingerprints.
3. Refuse stale or incomplete native output.

Important criteria:
- These tests load specifications only; they never run the optimizer.

Inputs/Outputs:
- Input: dimensions and synthetic native parameter echoes.
- Output: deterministic specification assertions.
"""

import hashlib
import json
import pytest
from benchmarks.algorithms.sa_specification import SaSpecification
from rko_solver.engines.rust.runner import NativeRunner
from rko_solver.orchestration.types import RunConfig


@pytest.mark.parametrize("dimension", [8, 100, 500, 1000])
def test_export_matches_the_production_loader(dimension: int) -> None:
    """Export the actual production vector rather than a native benchmark's old grid.

    Args:
        dimension (int): Problem dimension used for factory validation.
    """
    config = RunConfig(control=0, pool_size=10, active_metaheuristics=("SA",))
    rows, _ = NativeRunner(config, dimension, None, None).specs()
    exported = SaSpecification.for_dimension(dimension)
    assert exported.values == tuple(rows[0][1])
    assert exported.values == (
        1027.0,
        340.3863933617792,
        0.8601360548373771,
        0.0035204893377555175,
        0.008242788398191462,
    )
    assert rows[0][0] == "SA" and rows[0][3] is False


def test_native_arguments_preserve_order_and_float_values() -> None:
    """Round-trip the exported numbers through the benchmark's textual argument."""
    specification = SaSpecification.for_dimension(100)
    arguments = specification.native_arguments()
    assert arguments[0] == "--sa-parameters"
    assert tuple(float(value) for value in arguments[1].split(",")) == specification.values
    assert len(specification.values) == 5


def test_fingerprint_covers_the_canonical_ordered_vector() -> None:
    """Keep equivalent exports identifiable and detect reordered parameters."""
    specification = SaSpecification.for_dimension(100)
    encoded = json.dumps(specification.values, separators=(",", ":"), allow_nan=False).encode()
    assert specification.sha256 == hashlib.sha256(encoded).hexdigest()
    assert specification.sha256 == SaSpecification.for_dimension(500).sha256
    assert specification.sha256 != SaSpecification(tuple(reversed(specification.values))).sha256


@pytest.mark.parametrize(
    "values",
    [(), (1000.0, 1000000.0, 0.99, 0.01, 0.05), (1027.0, float("nan"), 0.86, 0.0035, 0.0082)],
)
def test_stale_or_incomplete_native_specifications_are_refused(values: tuple[float, ...]) -> None:
    """Reject an echo mismatch before its solve can count as production evidence.

    Args:
        values (tuple[float, ...]): Invalid native parameter vector.
    """
    with pytest.raises(ValueError, match="specification"):
        SaSpecification.for_dimension(100).verify(values)


def test_an_exact_native_echo_is_accepted() -> None:
    """Allow an exactly matching parameter echo."""
    specification = SaSpecification.for_dimension(100)
    specification.verify(specification.values)


@pytest.mark.parametrize("dimension", [0, -1])
def test_invalid_dimensions_are_refused(dimension: int) -> None:
    """Reject invalid export requests instead of fabricating a native vector.

    Args:
        dimension (int): Invalid variable count.
    """
    with pytest.raises(ValueError, match="positive"):
        SaSpecification.for_dimension(dimension)
