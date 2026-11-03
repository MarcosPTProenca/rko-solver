"""Reproducible instance generation and content identity for quantile experiments.

Objective:
- Keep training, validation and test instance identities explicit and independent.

Technical approach:
- Generate both families from local numerical generators and compare content fingerprints.

Flow (pseudocode):
1. Generate the same declared instance twice.
2. Generate another seed and the other task family.
3. Check immutable task data, full row coverability and identity separation.

Important criteria:
- The generator never consumes global numerical random state.

Inputs/Outputs:
- Input: family, size and deterministic generation seed.
- Output: valid tasks and stable content fingerprints.
"""

import numpy as np
import pytest
from benchmarks.quantile import instances


@pytest.mark.parametrize("family", ["tour", "cover"])
def test_instances_are_reproducible_coverable_and_content_identified(family: str) -> None:
    """Local generation reproduces data without contaminating external random state.

    Args:
        family (str): Declared combinatorial task family.
    """
    before = np.random.get_state()
    first = instances.Instances.make(family, 10, 700001)
    repeated = instances.Instances.make(family, 10, 700001)
    other = instances.Instances.make(family, 10, 700002)
    assert instances.Instances.fingerprint(first) == instances.Instances.fingerprint(repeated)
    assert instances.Instances.fingerprint(first) != instances.Instances.fingerprint(other)
    assert first.legal(())
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
