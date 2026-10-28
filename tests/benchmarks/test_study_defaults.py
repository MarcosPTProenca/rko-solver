"""Bindings between packaged SA defaults, the tuning harness baseline and archived evidence.

Objective:
- Keep the packaged fixed simulated-annealing preset identical to the configuration validated in
  the archived study, while the harness baseline stays the default the studies compared against.

Technical approach:
- Read the sealed reference and the packaged TOML through their public loaders and compare exact
  values, so rounding or an unreviewed edit fails.

Flow (pseudocode):
1. Load the archived locked configuration and archived default.
2. Compare them with the packaged offline values and the harness baseline.
3. Check unchanged algorithms still follow their packaged defaults.

Important criteria:
- The packaged SA preset must equal the validated candidate exactly.
- The harness baseline must not move when the packaged default changes.

Inputs/Outputs:
- Input: the committed reference and packaged parameter files.
- Output: assertions only.
"""

import json
import pytest
from benchmarks.tuning.spaces import SPACES
from pathlib import Path
from rko_solver.parameters.loader import load_offline

REFERENCE = Path(__file__).resolve().parents[2] / "benchmarks" / "general-sa-preset-reference.json"
HISTORICAL_SA = {
    "sa_max": 1000.0,
    "t0": 1000000.0,
    "alpha": 0.99,
    "beta_min": 0.01,
    "beta_max": 0.05,
}


def test_packaged_sa_offline_equals_the_validated_candidate() -> None:
    """The shipped fixed preset is exactly the configuration locked before confirmation."""
    data = json.loads(REFERENCE.read_text(encoding="utf-8"))
    locked = data["selection"]["config"]
    assert data["selection"]["candidate"] == "2c10f3beab1c2e7b"
    assert load_offline("SA") == locked


def test_harness_baseline_stays_the_default_the_studies_compared_against() -> None:
    """The tuning baseline is independent of the packaged default and matches the archive."""
    data = json.loads(REFERENCE.read_text(encoding="utf-8"))
    archived = data["protocol"]["candidates"]["default"]
    assert archived == HISTORICAL_SA
    assert SPACES["SA"].default() == HISTORICAL_SA


@pytest.mark.parametrize("algorithm", sorted(set(SPACES) - {"SA"}))
def test_unchanged_algorithms_follow_their_packaged_defaults(algorithm: str) -> None:
    """Only simulated annealing has a pinned historical baseline.

    Args:
        algorithm (str): Tuned algorithm other than SA.
    """
    packaged = load_offline(algorithm)
    space = SPACES[algorithm]
    assert space.default() == {spec.name: packaged[spec.name] for spec in space.specs}
